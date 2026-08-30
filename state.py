"""
state.py — Person 2: Conversation State & Memory.

Implements the Person 2 contract from interfaces.py:

    state  = new_state(session_id, user_profile)        # call from agent.reset()
    state  = update_state(state, user_message)            # call each turn (non-mutating)
    count  = estimate_result_count(state, catalog)       # over-generality detection
    summary = summarize(state)                            # context distillation (one line)
    compact = distill(state)                              # context distillation (JSON-able dict)

What update_state() does per message (regex-first, LLM fallback):

    1. Intent routing        — BUYING / BROWSING via intent.classify_intent()
    2. Information accumulation — parse constraints into slots
    3. Intent override       — "ignore my earlier preference" reverts the turn-1
                               disclosure, then applies the new requirement
    4. Rejection tracking    — "not that one" adds shown ASINs to rejected_asins
    5. No-preference memory  — records attributes the user said they don't care
                               about, so Person 3 never re-asks them

Constraint → slot mapping mirrors the evaluator's own classify_constraint()
(evaluator/local_evaluator.py) so a disclosed constraint lands in the slot the
evaluator would classify it under:

    "budget around $45.99"  → price_max = 45.99
    "cotton"                → material = "cotton"
    "color: blue"           → color = "blue"
    "Size: 10.5 wide"      → size = ...
    "Department: womens"   → gender = "women"
    "running / hiking / winter / gym / outdoor / work" → use_case
    everything else         → features.append(raw)

Override semantics (the hardest scenario): the simulator's override replaces the
turn-1 soft preference (behavior.override.old_value) with the hard constraint
(behavior.override.new_value). Constraints disclosed via ask_attribute replies
between turn 1 and the override stay valid — the evaluator's `disclosed` set
persists across the override. So update_state() reverts ONLY the turn-1
disclosure and keeps everything accumulated since. Reverting is
source-tracked: a slot is only cleared if the turn-1 disclosure is still the
latest source of that slot (if the user re-affirmed the same slot later, it
stays).

update_state() NEVER raises — the evaluator counts exceptions in respond() as a
miss, so every parsing failure degrades to a no-op.
"""

from __future__ import annotations

import copy
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field, replace
from typing import Any

from intent import MessageKind, classify_intent, detect_message_type, normalize_text
from interfaces import ConversationState, Slots


# =============================================================================
# Vocabularies — mirrored from evaluator/local_evaluator.py so slot mapping
# matches the evaluator's own classification of the same strings.
# =============================================================================

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
_MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)
_SIZE_WORDS = ("size", "sizing", "width", "wide", "narrow")
_STYLE_WORDS = ("department", "style", "fit", "sleeve", "neck")
_USE_CASE_WORDS = ("hiking", "running", "gym", "winter", "outdoor", "work")
_BUDGET_RE = re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)")
_GENDER_VALUE_RE = re.compile(r"^department\s*:\s*(.+)$", re.I)
_BRAND_VALUE_RE = re.compile(r"^brand\s*:\s*(.+)$", re.I)
_SIZE_VALUE_RE = re.compile(r"^(?:size|sizing|width)\s*:\s*(.+)$", re.I)

# ask_attribute names the user may say they have no preference for — mirrors
# evaluator ALLOWED_ATTRIBUTES.
ALLOWED_ATTRIBUTES = {
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
}

_ATTR_TO_SLOTS = {
    "budget": ["price_max", "price_min"],
    "feature": ["features"],
}


# Messages that carry no positive search signal. update_state keeps the
# previous last_query on these so the reranker preference query is not
# polluted with negation noise ("i don't have a preference for budget").
_META_MESSAGE_KINDS = {
    MessageKind.NO_PREFERENCE,
    MessageKind.BOUNDARY_NO_PREFERENCE,
    MessageKind.REJECTION,
}



# =============================================================================
# Constraint → slot mapping
# =============================================================================

def _distinct_words(words: list[str]) -> list[str]:
    """Lowercase, dedupe, preserve first-seen order."""
    seen: list[str] = []
    for word in words:
        lowered = word.lower()
        if lowered not in seen:
            seen.append(lowered)
    return seen


def constraint_to_slots(constraint: str) -> tuple[dict[str, Any], list[str]]:
    """Map one disclosed constraint string to slot assignments.

    Returns (scalar_assignments, feature_adds):
        scalar_assignments — {"color": "blue"} / {"price_max": 45.99} / {"gender": "women"} ...
        feature_adds        — raw strings to append to slots.features

    Mirrors evaluator classify_constraint() ordering: budget → material → color
    → size → department(gender) → brand → use_case → style → feature.
    Department is special-cased into the gender slot (the evaluator's rerank
    treats details["Department"] as the structural gender field, so the slot is
    more useful than a generic style string there).
    """
    raw = normalize_text(constraint or "")
    if not raw:
        return {}, []
    low = raw.lower()

    # budget: "budget around $45.99", "under $50", "<= $50"
    if "budget" in low or re.search(r"(?:\$|<=|under)\s*\d", low):
        if m := _BUDGET_RE.search(low):
            price = float(m.group(1))
            # "at least / minimum $X" sets a floor; everything else a ceiling
            if "at least" in low or "minimum" in low or "more than" in low:
                return {"price_min": price}, []
            return {"price_max": price}, []
        return {}, [raw]

    materials = _distinct_words(_MATERIAL_RE.findall(raw))
    if len(materials) >= 2:
        return {"material": materials[0]}, [raw]
    if materials:
        return {"material": materials[0]}, []

    if m := _GENDER_VALUE_RE.match(low):
        value = m.group(1).strip()
        gender = "women" if value.startswith("women") or value.startswith("womens") or value.startswith("female") else (
            "men" if value.startswith("men") or value.startswith("mens") or value.startswith("male") else value
        )
        return {"gender": gender}, []
    if m := _BRAND_VALUE_RE.match(low):
        return {"brand": m.group(1).strip().title()}, []

    colors = _distinct_words(_COLOR_RE.findall(raw))
    if len(colors) >= 2:
        return {"color": colors[0]}, [raw]
    if colors:
        return {"color": colors[0]}, []
    if m := re.match(r"^color\s*:\s*(.+)$", low):
        return {"color": m.group(1).strip()}, []

    if any(word in low for word in _SIZE_WORDS):
        if m := _SIZE_VALUE_RE.match(low):
            return {"size": m.group(1).strip()}, []
        return {}, [raw]

    if any(word in low for word in _USE_CASE_WORDS):
        for word in _USE_CASE_WORDS:
            if word in low:
                return {"use_case": word}, []

    if any(word in low for word in _STYLE_WORDS):
        if m := re.match(r"^(?:style|fit)\s*:\s*(.+)$", low):
            return {"style": m.group(1).strip()}, []
        return {}, [raw]

    return {}, [raw]


# =============================================================================
# State objects
# =============================================================================

@dataclass
class Disclosure:
    """One constraint-bearing event, kept for override reversion and debugging.

    `slots` holds the scalar assignments made from this disclosure (e.g.
    {"color": "blue"}); `feature_adds` the raw strings appended to features.
    """
    id: int
    turn: int
    kind: str
    raw: str
    slots: dict[str, Any] = field(default_factory=dict)
    feature_adds: list[str] = field(default_factory=list)


@dataclass
class ManagedState(ConversationState):
    """ConversationState + Person 2 provenance bookkeeping.

    Passes as a plain ConversationState to Persons 1 and 3 (isinstance check
    passes), so nothing downstream needs to change. Extra fields:

      disclosures      — list[Disclosure], every constraint applied so far
      slot_sources     — {slot_name: disclosure_id} that last set it
      feature_sources  — {feature_string: disclosure_id} that added it
      no_preference    — attributes the user said they have no preference for
    """

    disclosures: list[Disclosure] = field(default_factory=list)
    slot_sources: dict[str, int] = field(default_factory=dict)
    feature_sources: dict[str, int] = field(default_factory=dict)
    no_preference: set[str] = field(default_factory=set)
    llm_fallback_used: bool = False


def new_state(session_id: str, user_profile: dict | None = None) -> ManagedState:
    """Fresh state for a session — call from Agent.reset()."""
    state = ManagedState()
    state.session_id = session_id
    state.user_profile = copy.deepcopy(user_profile or {})
    return state


def as_managed(state: ConversationState) -> ManagedState:
    """Upgrade a plain ConversationState (e.g. built by Person 3) to ManagedState."""
    if isinstance(state, ManagedState):
        return state
    upgraded = ManagedState(
        session_id=state.session_id,
        turn_count=state.turn_count,
        max_turns=state.max_turns,
        user_profile=state.user_profile,
        intent=state.intent,
        slots=replace(state.slots),
        history=copy.deepcopy(state.history),
        asked_clarifications=set(state.asked_clarifications),
        rejected_asins=list(state.rejected_asins),
        last_query=state.last_query,
        last_candidates=list(state.last_candidates),
    )
    upgraded.slots.features = list(state.slots.features)
    return upgraded


def _copy_state(state: ManagedState) -> ManagedState:
    """Deep copy for non-mutating update_state()."""
    clone = replace(state)
    clone.slots = replace(state.slots)
    clone.slots.features = list(state.slots.features)
    clone.history = copy.deepcopy(state.history)
    clone.disclosures = list(state.disclosures)
    clone.slot_sources = dict(state.slot_sources)
    clone.feature_sources = dict(state.feature_sources)
    clone.no_preference = set(state.no_preference)
    clone.asked_clarifications = set(state.asked_clarifications)
    clone.rejected_asins = list(state.rejected_asins)
    clone.last_candidates = list(state.last_candidates)
    return clone


# =============================================================================
# Core update logic
# =============================================================================

def _next_disclosure_id(state: ManagedState) -> int:
    return state.disclosures[-1].id + 1 if state.disclosures else 0


def _apply_disclosure(state: ManagedState, turn: int, kind: str, raw: str,
                      scalar_assignments: dict[str, Any], feature_adds: list[str]) -> Disclosure:
    """Record one disclosure and apply its assignments to the slots."""
    disc = Disclosure(
        id=_next_disclosure_id(state), turn=turn, kind=kind, raw=raw,
        slots=dict(scalar_assignments), feature_adds=list(feature_adds),
    )
    for slot, value in scalar_assignments.items():
        if not hasattr(state.slots, slot):
            continue
        existing = getattr(state.slots, slot)
        if (
            existing is not None
            and existing != value
            and slot in ("material", "color")
        ):
            # Blends get re-disclosed with different primaries ("polyester"
            # then "95% cotton, 5% spandex" describe the SAME product). Keep
            # the superseded value as a feature so no disclosed constraint is
            # ever lost from the state.
            preserved = str(existing)
            if preserved not in state.slots.features:
                state.slots.features.append(preserved)
                state.feature_sources[preserved] = disc.id
        setattr(state.slots, slot, value)
        state.slot_sources[slot] = disc.id
    for feature in feature_adds:
        if feature not in state.slots.features:
            state.slots.features.append(feature)
            state.feature_sources[feature] = disc.id
    state.disclosures.append(disc)
    return disc


def _revert_disclosure(state: ManagedState, disc: Disclosure) -> None:
    """Undo a disclosure's slot assignments — only where it is still the latest
    source. Slots re-affirmed by a later disclosure are kept."""
    for slot, _value in disc.slots.items():
        if state.slot_sources.get(slot) == disc.id:
            setattr(state.slots, slot, None)
            del state.slot_sources[slot]
    for feature in disc.feature_adds:
        if state.feature_sources.get(feature) == disc.id:
            if feature in state.slots.features:
                state.slots.features.remove(feature)
            del state.feature_sources[feature]


def _current_turn(state: ManagedState) -> int:
    """The evaluator passes the turn number via agent.respond(); Person 3 is
    told to set state.turn_count before calling update_state(). Fall back to
    history length so the module works standalone too."""
    return state.turn_count if state.turn_count > 0 else len(state.history) + 1


def update_state(state: ConversationState, message: str) -> ManagedState:
    """Person 2 contract — update state with a new user message.

    Non-mutating: returns a new state; the input state is left untouched.
    Never raises: unparseable messages become no-ops (with history recorded).
    """
    s = _copy_state(as_managed(state))
    turn = _current_turn(s)
    raw_message = message or ""
    s.history.append({"role": "user", "content": raw_message})
    try:
        kind, fields = detect_message_type(raw_message)
        s.intent = classify_intent(raw_message, s.intent)
        if kind not in _META_MESSAGE_KINDS:
            # Content-bearing message. rerank() appends last_query to its
            # preference query to catch signals ("cozy", "a special gift")
            # not yet parsed into slots. Meta messages carry no positive
            # signal, so they keep the previous query instead.
            s.last_query = raw_message

        if kind is MessageKind.INITIAL_BUYING:
            if fields.get("category"):
                s.slots.category = fields["category"]
            if fields.get("constraint"):
                scalars, feats = constraint_to_slots(fields["constraint"])
                _apply_disclosure(s, turn, "initial_hard", fields["constraint"], scalars, feats)

        elif kind is MessageKind.INITIAL_BROWSING:
            if fields.get("category"):
                s.slots.category = fields["category"]

        elif kind is MessageKind.INITIAL_OVERRIDE:
            if fields.get("category"):
                s.slots.category = fields["category"]
            if fields.get("constraint"):
                scalars, feats = constraint_to_slots(fields["constraint"])
                _apply_disclosure(s, turn, "initial_soft", fields["constraint"], scalars, feats)

        elif kind is MessageKind.OVERRIDE:
            value = fields.get("value")
            if value:
                # Revert the turn-1 (soft) preference — it is the earliest
                # disclosure and the one the user is overriding. Constraints
                # accumulated via later ask_attribute replies stay valid.
                if s.disclosures:
                    _revert_disclosure(s, s.disclosures[0])
                scalars, feats = constraint_to_slots(value)
                _apply_disclosure(s, turn, "override", value, scalars, feats)

        elif kind is MessageKind.ANSWER:
            items = [item.strip() for item in (fields.get("items") or "").split(";") if item.strip()]
            for item in items:
                scalars, feats = constraint_to_slots(item)
                _apply_disclosure(s, turn, "answer", item, scalars, feats)

        elif kind is MessageKind.NO_PREFERENCE:
            attribute = (fields.get("attribute") or "").strip().lower()
            if attribute:
                s.no_preference.add(attribute)
                if attribute in ALLOWED_ATTRIBUTES:
                    # Record the attribute AND its slot names. The
                    # orchestrator checks slot names ("price_max") when
                    # deciding what is still missing; without the mapped
                    # slots a declined budget would be re-asked forever.
                    s.asked_clarifications.add(attribute)
                    for slot in _ATTR_TO_SLOTS.get(attribute, []):
                        s.asked_clarifications.add(slot)

        elif kind is MessageKind.BOUNDARY_NO_PREFERENCE:
            attribute = (fields.get("attribute") or "").strip().lower()
            if attribute:
                s.no_preference.add(attribute)
                if attribute in ALLOWED_ATTRIBUTES:
                    # Record the attribute AND its slot names. The
                    # orchestrator checks slot names ("price_max") when
                    # deciding what is still missing; without the mapped
                    # slots a declined budget would be re-asked forever.
                    s.asked_clarifications.add(attribute)
                    for slot in _ATTR_TO_SLOTS.get(attribute, []):
                        s.asked_clarifications.add(slot)

        elif kind is MessageKind.REJECTION:
            # Real-user insurance: the evaluator never sends these. Reject the
            # top of the last shown candidate list.
            for product in (s.last_candidates or [])[:3]:
                asin = product.get("parent_asin") if isinstance(product, dict) else None
                if asin and asin not in s.rejected_asins:
                    s.rejected_asins.append(asin)

        elif kind is MessageKind.UNKNOWN:
            _fallback_extract(s, raw_message, turn)

    except Exception:
        # Never raise inside the evaluator loop — an exception in respond()
        # counts as a miss. Degrade to history-only.
        pass

    return s


def _fallback_extract(s: ManagedState, message: str, turn: int) -> None:
    """Unknown template: heuristic extraction first, then optional LLM.

    The heuristic applies the same constraint regexes to the raw message. If an
    LLM key is configured (TECHJAM_LLM_API_KEY / OPENAI_API_KEY), the LLM gets
    the message plus current slots and returns strict JSON; its assignments
    are applied only for slots not already set.
    """
    text = normalize_text(message)
    if m := re.search(r"looking for ([^.]+)", text, re.I):
        if not s.slots.category:
            s.slots.category = m.group(1).strip().rstrip(",")

    scalars, feats = constraint_to_slots(text)
    if scalars or feats:
        _apply_disclosure(s, turn, "heuristic", text, scalars, feats)
        return

    llm = _llm_extract_slots(text, s)
    if llm:
        s.llm_fallback_used = True
        assignments = {}
        feature_adds = []
        for slot, value in llm.items():
            if slot == "features" and isinstance(value, list):
                feature_adds = [str(v) for v in value if v]
            elif hasattr(s.slots, slot) and value is not None and getattr(s.slots, slot) is None:
                assignments[slot] = value
        if assignments or feature_adds:
            _apply_disclosure(s, turn, "llm", text, assignments, feature_adds)


# =============================================================================
# Optional LLM fallback (env-keyed, never committed — no key, no call)
# =============================================================================

_LLM_MODEL = os.environ.get("TECHJAM_LLM_MODEL", "gpt-4o-mini")
_LLM_BASE = os.environ.get("TECHJAM_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
_LLM_KEY = os.environ.get("TECHJAM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")

_LLM_SYSTEM_PROMPT = (
    "You extract shopping constraints from a shopper's message for a clothing/shoes/jewelry "
    "catalog agent. Reply with ONLY a JSON object. Allowed keys: category, brand, gender, "
    "color, material, style, size, use_case, price_min, price_max, features (list of strings). "
    "Include a key only if the message states it. Omit everything else. "
    'Example: {"color":"blue","price_max":100}'
)


def _llm_extract_slots(message: str, state: ManagedState) -> dict | None:
    """OpenAI-compatible chat completion with JSON mode. Returns parsed JSON or
    None on any failure (missing key, timeout, bad JSON). 8s timeout."""
    if not _LLM_KEY or not message:
        return None
    payload = {
        "model": _LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM_PROMPT},
            {"role": "user", "content": f"Current known slots: {json.dumps(state.slots.to_dict())}\nMessage: {message}"},
        ],
    }
    try:
        request = urllib.request.Request(
            f"{_LLM_BASE}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_LLM_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


# =============================================================================
# estimate_result_count — Person 3 uses this for over-generality detection
# =============================================================================

def estimate_result_count(state: ConversationState, catalog: Any) -> int:
    """Estimate how many catalog products the current category would match.

    catalog must expose `products` (list) and `category_index`
    (dict[str, list[int]] — lowercase category string → product indices), which
    Catalog (catalog.py) does.

    Matching strategy, in order:
      1. no category set          → whole catalog size
      2. exact category key       → its product count
      3. token intersection       — e.g. "women clothing" → products indexed
                                    under BOTH "women" AND "clothing"
      4. substring fallback       — any key containing any token; largest count
    Never returns 0 (Person 3 branches on > threshold only).
    """
    try:
        products = getattr(catalog, "products", None) or []
        category_index = getattr(catalog, "category_index", None) or {}
        category = (state.slots.category or "").strip().lower()
        if not category:
            return len(products)

        if category in category_index:
            return len(category_index[category])

        tokens = [t for t in re.findall(r"[a-z0-9&]+", category) if t]
        found = [category_index[t] for t in tokens if t in category_index]
        if found:
            intersection = set(found[0])
            for indices in found[1:]:
                intersection &= set(indices)
            # A partial token match that empties out (e.g. a token that is a
            # strict subset path) falls back to the largest single-token count
            if intersection:
                return len(intersection)
            return max(len(indices) for indices in found)

        counts = [
            len(indices) for key, indices in category_index.items()
            if any(token in key for token in tokens)
        ]
        return max(counts) if counts else len(products)
    except Exception:
        return 1


# =============================================================================
# Context distillation — compact representations for LLM prompts
# =============================================================================

def distill(state: ConversationState) -> dict:
    """Compact, JSON-serializable view of everything the agent knows.

    Use in place of raw history in LLM prompts (context distillation): the
    conversation can be 10 turns; this stays ~15 keys.
    """
    managed = as_managed(state)
    slots = managed.slots.to_dict()
    return {
        "intent": managed.intent,
        "turn": managed.turn_count,
        "slots": slots,
        "no_preference": sorted(managed.no_preference),
        "asked_about": sorted(managed.asked_clarifications),
        "rejected_count": len(managed.rejected_asins),
        "profile_summary": (managed.user_profile or {}).get("summary"),
        "preference_tags": (managed.user_profile or {}).get("preference_tags"),
        "summary": summarize(managed),
    }


def summarize(state: ConversationState) -> str:
    """One-line natural language summary of the user's current requirements."""
    managed = as_managed(state)
    parts: list[str] = []
    slots = managed.slots
    if slots.category:
        parts.append(f"shopping for {slots.category}")
    bits: list[str] = []
    if slots.brand:
        bits.append(f"brand {slots.brand}")
    if slots.gender:
        bits.append(f"for {slots.gender}")
    if slots.color:
        bits.append(f"{slots.color}")
    if slots.material:
        bits.append(f"{slots.material}")
    if slots.style:
        bits.append(f"{slots.style} style")
    if slots.size:
        bits.append(f"size {slots.size}")
    if slots.use_case:
        bits.append(f"for {slots.use_case}")
    if slots.price_max is not None:
        bits.append(f"under ${slots.price_max:g}")
    if bits:
        parts.append("wants " + ", ".join(bits))
    if slots.features:
        parts.append("features: " + "; ".join(slots.features))
    no_pref = sorted(managed.no_preference)
    if no_pref:
        parts.append("no preference for " + ", ".join(no_pref))
    return " | ".join(parts) if parts else "no requirements captured yet"
