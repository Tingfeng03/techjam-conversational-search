"""
exploration/test_state_replay.py — Person 2 end-to-end verification.

Replays update_state() against the evaluator's OWN message generator
(evaluator/local_evaluator.py) for all 200 public sessions — the exact
message stream the evaluator will send at scoring time — and measures:

  1. Constraint coverage — every constraint the simulator disclosed is
     reflected in the state's slots (or features) at the end of the session.
  2. Override correctness — for intent_override sessions: the old turn-1
     preference is retracted and the new hard constraint is present after
     the override message.
  3. Intent accuracy — buying sessions classified BUYING, browsing/boundary
     classified BROWSING, override sessions BUYING after the override.

The ask policy is deliberately dumb (round-robin over attributes the user
hasn't declined) — it exists only to drive the conversation; quality of the
asks is Person 3's problem.

Run from repo root:
    python3 exploration/test_state_replay.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluator.local_evaluator import (
    behavior_for,
    classify_constraint,
    coarse_category,
    customer_reply,
    initial_message,
    intent_card,
    load_jsonl,
    materialize_hidden_fields,
)
import random

from state import constraint_to_slots, estimate_result_count, new_state, update_state

MAX_TURNS = 10
ASK_ORDER = ["material", "color", "size", "style", "use_case", "budget", "feature", "brand", "other"]


def build_catalog_reference(path: str = "data/catalog.jsonl"):
    """Lightweight catalog: products + category_index (no models loaded)."""
    products: dict[str, dict] = {}
    categories: dict[str, list[str]] = {}
    category_index: dict[str, list[int]] = {}
    for idx, product in enumerate(load_jsonl(path)):
        asin = str(product["parent_asin"])
        products[asin] = product
        cats = [str(c) for c in product.get("categories") or []]
        categories[asin] = cats
        for cat in cats:
            key = cat.lower().strip()
            if key:
                category_index.setdefault(key, []).append(idx)

    class _Ref:
        pass

    ref = _Ref()
    ref.products = list(products.values())
    ref.category_index = category_index
    return products, categories, ref


def session_message_stream(sample: dict, products, categories):
    """Yield (turn, message) exactly as the evaluator would produce them,
    following a dumb round-robin ask policy."""
    card, behavior = materialize_hidden_fields(sample, products)
    eff = {**sample, "intent_card": card, "behavior": behavior}
    target = str(sample["ground_truth"]["parent_asin"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)
    asked: set[str] = set()
    turn = 0
    while turn < MAX_TURNS:
        turn += 1
        yield turn, message, disclosed, eff, override_applied
        if turn == MAX_TURNS:
            break
        override = eff.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            ask = next((a for a in ASK_ORDER if a not in asked), "other")
            asked.add(ask)
            message, boundary_used = customer_reply(eff, ask, disclosed, boundary_used)


def _semantic_covered(state, constraint: str) -> bool:
    """Is this disclosed constraint's information recoverable from the state?

    Checks, in order:
      1. scalar path     — the mapped slot holds the mapped value
      2. preserved path  — a superseded value was preserved as a feature
      3. feature path    — the constraint (or its mapped feature) is stored
      4. piecewise path  — the constraint arrived inside an ANSWER whose "; "
                           split broke it into pieces that were all stored
    """
    scalars, feats = constraint_to_slots(constraint)
    feature_text = " || ".join(state.slots.features)
    if scalars and all(getattr(state.slots, s, None) == v for s, v in scalars.items()):
        return True
    if scalars and all(str(v).lower() in feature_text.lower() for v in scalars.values()):
        return True
    if feats and all(f in state.slots.features for f in feats):
        return True
    pieces = [p.strip() for p in constraint.split(";") if p.strip()]
    if pieces and all(
        any(p.lower() == f.lower() or p.lower() in f.lower() for f in state.slots.features)
        for p in pieces
    ):
        return True
    if scalars and all(str(v).lower() in feature_text.lower() for v in scalars.values()):
        return True
    return False


def coverage_of(state, constraints: list[str]) -> tuple[int, int]:
    """How many disclosed constraints are reflected in the final state."""
    hit = sum(1 for c in constraints if _semantic_covered(state, c))
    return hit, len(constraints)


def main() -> None:
    products, categories, catalog_ref = build_catalog_reference()
    samples = load_jsonl("data/public_set.jsonl")
    print(f"[replay] {len(samples)} sessions, catalog {len(products):,} products\n")

    totals = Counter()
    cov_hits = cov_total = 0
    intent_correct = 0
    override_checked = override_ok = 0
    estimate_calls = 0

    for sample in samples:
        scenario = sample["scenario_type"]
        state = new_state(f"replay_{sample['sample_id']}", sample.get("user_profile") or {})
        overridden_constraints: list[str] = []
        post_override_checks: dict[int, tuple] = {}

        stream = session_message_stream(sample, products, categories)
        for turn, message, disclosed_before, eff, override_applied in stream:
            state.turn_count = turn
            state = update_state(state, message)

            # after the override message lands, check reversion
            if scenario == "intent_override" and message.startswith("Actually, ignore my earlier preference"):
                old_value = str(eff["behavior"]["override"]["old_value"])
                new_value = str(eff["behavior"]["override"]["new_value"])
                old_scalars, old_feats = constraint_to_slots(old_value)
                new_scalars, new_feats = constraint_to_slots(new_value)
                # Retraction = the initial disclosure no longer sources the
                # slot (value may legitimately reappear via the new requirement).
                retracted = all(
                    getattr(state.slots, slot, None) != value
                    or getattr(state, "slot_sources", {}).get(slot) != 0
                    for slot, value in old_scalars.items()
                ) and all(f not in state.slots.features for f in old_feats)
                present = all(
                    getattr(state.slots, slot, None) == value for slot, value in new_scalars.items()
                ) and all(f in state.slots.features for f in new_feats)
                post_override_checks[turn] = (retracted, present)
                overridden_constraints = [old_value]

            if turn == 2:
                _ = estimate_result_count(state, catalog_ref)
                estimate_calls += 1

        # end of session — measure coverage over what the user actually told us,
        # excluding the overridden (retracted) old preference
        _, _, final_disclosed, eff, _ = list(session_message_stream(sample, products, categories))[-1]
        active = [c for c in final_disclosed if c not in overridden_constraints]
        hits, total = coverage_of(state, active)
        cov_hits += hits
        cov_total += total

        expected = "BUYING" if scenario == "buying" else "BROWSING"
        if scenario == "intent_override":
            expected = "BUYING"
        if state.intent == expected:
            intent_correct += 1

        if post_override_checks:
            override_checked += 1
            if all(retracted and present for retracted, present in post_override_checks.values()):
                override_ok += 1

        totals[scenario] += 1

    n = len(samples)
    print(f"scenario counts        : {dict(totals)}")
    print(f"constraint coverage    : {cov_hits}/{cov_total} = {cov_hits / cov_total:.1%}")
    print(f"override reversion     : {override_ok}/{override_checked} correct")
    print(f"intent accuracy        : {intent_correct}/{n} = {intent_correct / n:.1%}")
    print(f"estimate_result_count  : {estimate_calls} calls, no exceptions")


if __name__ == "__main__":
    main()
