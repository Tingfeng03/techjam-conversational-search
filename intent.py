"""
intent.py — Person 2: message understanding.

The evaluator's customer simulator is deterministic and template-based
(evaluator/local_evaluator.py). Every message it sends matches one of a small
number of fixed templates:

    buying turn 1   "I'm looking for {category}. A key requirement is: {constraint}."
    override turn 1 "I'm looking for {category}. {old soft preference}"
    browsing turn 1 "I'm looking for {category}, but I'm still exploring."
    override (3/4)  "Actually, ignore my earlier preference. What I need is: {new}."
    ask reply       "For that, what matters is: {c1}; {c2}."
    ask reply (no)  "I don't have an additional preference for {attribute}."
    ask reply (noQ) "Those options are not quite right yet. Ask me about one specific attribute."
    boundary reply  "I don't have a preference for {attribute}; please use your judgment."

detect_message_type() identifies which template a message matches and extracts
its fields (category, constraint values, attribute names). classify_intent()
maps that to BUYING / BROWSING. Regex-first: exact template matches are free and
deterministic; messages that match no template return UNKNOWN and fall through
to state.py's heuristic + optional LLM path.
"""

from __future__ import annotations

import re
from enum import Enum


class MessageKind(str, Enum):
    INITIAL_BUYING = "initial_buying"
    INITIAL_OVERRIDE = "initial_override"
    INITIAL_BROWSING = "initial_browsing"
    OVERRIDE = "override"
    ANSWER = "answer"
    NO_PREFERENCE = "no_preference"
    BOUNDARY_NO_PREFERENCE = "boundary_no_preference"
    NO_ANSWER = "no_answer"
    REJECTION = "rejection"
    UNKNOWN = "unknown"


# Turn-1 messages always start "I'm looking for ..." — apostrophes may arrive
# as ASCII ' or curly ’, so normalize before matching.
_INITIAL_BUYING_RE = re.compile(
    r"^I'?m looking for (?P<category>[^.]+)\.\s*A key requirement is:\s*(?P<constraint>.+?)\.?\s*$"
)
_INITIAL_BROWSING_RE = re.compile(
    r"^I'?m looking for (?P<category>.+?),\s*but I'?m still exploring\.?\s*$"
)
# Generic turn-1 form: category sentence followed by an optional soft-preference
# sentence. The override scenario uses this shape on turn 1.
_INITIAL_OPEN_RE = re.compile(
    r"^I'?m looking for (?P<category>[^.]+?)\.(?:\s+(?P<rest>.+))?$"
)
_OVERRIDE_RE = re.compile(
    r"(?:ignore|forget|disregard|never mind).{0,40}?(?:preference|earlier|before)",
    re.I | re.S,
)
_OVERRIDE_VALUE_RE = re.compile(r"What I need is:\s*(?P<value>.+?)\.?\s*$", re.I)
_ANSWER_RE = re.compile(r"what matters is:\s*(?P<items>.+?)\.?\s*$", re.I)
_NO_PREFERENCE_RE = re.compile(
    r"^I don'?t have an? (?:additional )?preference for (?P<attribute>[^.;]+?)\.?\s*$",
    re.I,
)
_BOUNDARY_RE = re.compile(
    r"^I don'?t have a preference for (?P<attribute>[^.;]+?);\s*please use your judgment\.?\s*$",
    re.I,
)
_NO_ANSWER_RE = re.compile(r"ask me about one specific attribute", re.I)
_REJECTION_RE = re.compile(
    r"\b(?:i )?(?:don'?t|do not|didn'?t) (?:want|like|need) (?:that|this|those|these|the)"
    r"|\bnot (?:that|this|those|these)\b"
    r"|\bno thanks\b"
    r"|\banything else\b"
    r"|\bshow me (?:something|anything) else\b",
    re.I,
)


def normalize_text(message: str) -> str:
    """Normalize quotes/whitespace so template regexes survive copy-paste artifacts."""
    text = (message or "").replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    return " ".join(text.split())


def detect_message_type(message: str) -> tuple[MessageKind, dict]:
    """Classify a user message into a MessageKind and extract its template fields.

    Returns (kind, fields) where fields may contain:
        category    — coarse category phrase (initial messages)
        constraint  — single constraint string (buying turn 1)
        value       — new requirement (override messages)
        items       — list of constraint strings (ask replies)
        attribute   — attribute name the user has no preference for
    """
    msg = normalize_text(message)

    if m := _INITIAL_BUYING_RE.match(msg):
        return MessageKind.INITIAL_BUYING, {
            "category": m.group("category").strip(),
            "constraint": m.group("constraint").strip(),
        }
    if m := _INITIAL_BROWSING_RE.match(msg):
        return MessageKind.INITIAL_BROWSING, {"category": m.group("category").strip()}
    if _OVERRIDE_RE.search(msg):
        value = None
        if m := _OVERRIDE_VALUE_RE.search(msg):
            value = m.group("value").strip()
        elif m := _ANSWER_RE.search(msg):
            value = m.group("items").strip()
        return MessageKind.OVERRIDE, {"value": value}
    if m := _ANSWER_RE.search(msg):
        return MessageKind.ANSWER, {"items": m.group("items").strip()}
    if m := _BOUNDARY_RE.match(msg):
        return MessageKind.BOUNDARY_NO_PREFERENCE, {"attribute": m.group("attribute").strip()}
    if m := _NO_PREFERENCE_RE.match(msg):
        return MessageKind.NO_PREFERENCE, {"attribute": m.group("attribute").strip()}
    if _NO_ANSWER_RE.search(msg):
        return MessageKind.NO_ANSWER, {}
    if m := _INITIAL_OPEN_RE.match(msg):
        rest = (m.group("rest") or "").strip()
        return MessageKind.INITIAL_OVERRIDE, {
            "category": m.group("category").strip(),
            "constraint": rest or None,
        }
    if _REJECTION_RE.search(msg):
        return MessageKind.REJECTION, {}
    return MessageKind.UNKNOWN, {}


def classify_intent(message: str, current: str = "UNKNOWN") -> str:
    """BUYING / BROWSING classification. Intent is sticky: it only changes on an
    explicit signal, never on a neutral reply.

    Signals (mirroring the simulator's scenarios):
      - "A key requirement is:"   → BUYING   (buying scenario, hard constraint)
      - "still exploring"         → BROWSING (browsing / boundary scenarios)
      - "What I need is:"         → BUYING   (override settles on a firm requirement)
    Anything else keeps the current value.
    """
    msg = normalize_text(message)
    if "a key requirement is:" in msg.lower():
        return "BUYING"
    if "still exploring" in msg.lower():
        return "BROWSING"
    if _OVERRIDE_VALUE_RE.search(msg):
        return "BUYING"
    return current if current in ("BUYING", "BROWSING") else "UNKNOWN"
