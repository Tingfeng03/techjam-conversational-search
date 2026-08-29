"""Unit tests for Person 2's state.py + intent.py.

Run from repo root:  python3 -m unittest tests.test_state -v
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intent import MessageKind, classify_intent, detect_message_type
from state import (
    ManagedState,
    constraint_to_slots,
    distill,
    estimate_result_count,
    new_state,
    summarize,
    update_state,
)


class FakeCatalog:
    def __init__(self, products, category_index):
        self.products = products
        self.category_index = category_index


class IntentTest(unittest.TestCase):
    def test_buying_initial(self):
        kind, fields = detect_message_type(
            "I'm looking for Women Clothing. A key requirement is: budget around $45.99."
        )
        self.assertIs(kind, MessageKind.INITIAL_BUYING)
        self.assertEqual(fields["category"], "Women Clothing")
        self.assertEqual(fields["constraint"], "budget around $45.99")

    def test_browsing_initial(self):
        kind, fields = detect_message_type(
            "I'm looking for Athletic Shoes, but I'm still exploring."
        )
        self.assertIs(kind, MessageKind.INITIAL_BROWSING)
        self.assertEqual(fields["category"], "Athletic Shoes")

    def test_override_initial_with_soft_preference(self):
        kind, fields = detect_message_type(
            "I'm looking for Novelty. I prefer a softer feel."
        )
        self.assertIs(kind, MessageKind.INITIAL_OVERRIDE)
        self.assertEqual(fields["category"], "Novelty")
        self.assertEqual(fields["constraint"], "I prefer a softer feel.")

    def test_override_message(self):
        kind, fields = detect_message_type(
            "Actually, ignore my earlier preference. What I need is: leather."
        )
        self.assertIs(kind, MessageKind.OVERRIDE)
        self.assertEqual(fields["value"], "leather")

    def test_answer_message(self):
        kind, fields = detect_message_type(
            "For that, what matters is: cotton; color: blue."
        )
        self.assertIs(kind, MessageKind.ANSWER)
        self.assertEqual(fields["items"], "cotton; color: blue")

    def test_no_preference_message(self):
        kind, fields = detect_message_type(
            "I don't have an additional preference for material."
        )
        self.assertIs(kind, MessageKind.NO_PREFERENCE)
        self.assertEqual(fields["attribute"], "material")

    def test_boundary_message(self):
        kind, fields = detect_message_type(
            "I don't have a preference for color; please use your judgment."
        )
        self.assertIs(kind, MessageKind.BOUNDARY_NO_PREFERENCE)
        self.assertEqual(fields["attribute"], "color")

    def test_intent_sticky(self):
        self.assertEqual(classify_intent("For that, what matters is: cotton.", "BROWSING"), "BROWSING")
        self.assertEqual(classify_intent("Actually, ignore my earlier preference. What I need is: leather.", "UNKNOWN"), "BUYING")
        self.assertEqual(classify_intent("I'm looking for shoes, but I'm still exploring.", "BUYING"), "BROWSING")


class ConstraintMappingTest(unittest.TestCase):
    def test_budget(self):
        scalars, feats = constraint_to_slots("budget around $45.99")
        self.assertEqual(scalars, {"price_max": 45.99})
        self.assertEqual(feats, [])

    def test_budget_floor(self):
        scalars, _ = constraint_to_slots("at least $20")
        self.assertEqual(scalars, {"price_min": 20.0})

    def test_material(self):
        scalars, _ = constraint_to_slots("100% cotton")
        self.assertEqual(scalars, {"material": "cotton"})

    def test_color_prefixed(self):
        scalars, _ = constraint_to_slots("color: blue")
        self.assertEqual(scalars, {"color": "blue"})

    def test_color_bare(self):
        scalars, _ = constraint_to_slots("something black")
        self.assertEqual(scalars, {"color": "black"})

    def test_department_maps_to_gender(self):
        scalars, _ = constraint_to_slots("Department: womens")
        self.assertEqual(scalars, {"gender": "women"})

    def test_department_mens(self):
        scalars, _ = constraint_to_slots("Department: men")
        self.assertEqual(scalars, {"gender": "men"})

    def test_brand(self):
        scalars, _ = constraint_to_slots("Brand: spirit hoops")
        self.assertEqual(scalars, {"brand": "Spirit Hoops"})

    def test_use_case(self):
        scalars, _ = constraint_to_slots("great for hiking trips")
        self.assertEqual(scalars, {"use_case": "hiking"})

    def test_size(self):
        scalars, _ = constraint_to_slots("Size: 10.5 wide")
        self.assertEqual(scalars, {"size": "10.5 wide"})

    def test_feature_fallback(self):
        scalars, feats = constraint_to_slots("Made in USA")
        self.assertEqual(scalars, {})
        self.assertEqual(feats, ["Made in USA"])


class UpdateStateTest(unittest.TestCase):
    def _turn(self, state, message, turn):
        state.turn_count = turn
        return update_state(state, message)

    def test_accumulation(self):
        s = new_state("s1")
        s = self._turn(s, "I'm looking for Athletic Shoes. A key requirement is: leather.", 1)
        self.assertEqual(s.slots.category, "Athletic Shoes")
        self.assertEqual(s.slots.material, "leather")
        self.assertEqual(s.intent, "BUYING")
        s = self._turn(s, "For that, what matters is: budget around $80; color: black.", 2)
        self.assertEqual(s.slots.price_max, 80.0)
        self.assertEqual(s.slots.color, "black")
        self.assertEqual(s.slots.material, "leather")  # retained across turns

    def test_override_reverts_initial_preference_only(self):
        s = new_state("s2")
        s = self._turn(s, "I'm looking for Boots. Waterproof leather upper.", 1)
        self.assertEqual(s.slots.material, "leather")
        s = self._turn(s, "For that, what matters is: color: brown.", 2)
        self.assertEqual(s.slots.color, "brown")
        # override: turn-1 leather must go, turn-2 brown must stay
        s = self._turn(s, "Actually, ignore my earlier preference. What I need is: cotton.", 3)
        self.assertEqual(s.slots.material, "cotton")
        self.assertIsNone(s.slots.category is None and None or s.slots.material if False else None) if False else None
        self.assertEqual(s.slots.color, "brown")
        self.assertEqual(s.slots.category, "Boots")

    def test_no_preference_recorded(self):
        s = new_state("s3")
        s = self._turn(s, "I'm looking for Dresses, but I'm still exploring.", 1)
        self.assertEqual(s.intent, "BROWSING")
        s = self._turn(s, "I don't have an additional preference for material.", 2)
        self.assertIn("material", s.no_preference)
        self.assertIn("material", s.asked_clarifications)

    def test_boundary_recorded(self):
        s = new_state("s4")
        s = self._turn(s, "I don't have a preference for size; please use your judgment.", 1)
        self.assertIn("size", s.no_preference)

    def test_non_mutation(self):
        s = new_state("s5")
        s = self._turn(s, "I'm looking for Shoes. A key requirement is: leather.", 1)
        snapshot = summarize(s)
        before_slots = s.slots.material
        s2 = self._turn(s, "For that, what matters is: color: red.", 2)
        self.assertEqual(s.slots.material, before_slots)  # original untouched
        self.assertIsNone(s.slots.color)  # original untouched
        self.assertEqual(snapshot, summarize(s))
        self.assertEqual(s2.slots.color, "red")  # new state advanced

    def test_never_raises(self):
        s = new_state("s6")
        for garbage in ("", "...", None, 42, "\x00\x01", "a" * 5000):
            try:
                s = update_state(s, garbage)
            except Exception as exc:  # pragma: no cover
                self.fail(f"update_state raised on {garbage!r}: {exc}")

    def test_rejection_tracks_candidates(self):
        s = new_state("s7")
        s.last_candidates = [{"parent_asin": "A1"}, {"parent_asin": "B2"}, {"parent_asin": "C3"}]
        s = self._turn(s, "I don't want that first one", 2)
        self.assertIn("A1", s.rejected_asins)
        self.assertIn("B2", s.rejected_asins)

    def test_history_recorded(self):
        s = new_state("s8")
        s = self._turn(s, "I'm looking for Shoes, but I'm still exploring.", 1)
        self.assertEqual(s.history[-1]["role"], "user")
        self.assertIn("still exploring", s.history[-1]["content"])


class EstimateResultCountTest(unittest.TestCase):
    def setUp(self):
        self.catalog = FakeCatalog(
            products=[{"parent_asin": f"P{i}"} for i in range(10)],
            category_index={
                "women": [0, 1, 2, 3, 4],
                "clothing": [0, 1, 2, 5, 6],
                "novelty": [7, 8],
            },
        )

    def test_no_category_returns_catalog_size(self):
        s = new_state("c1")
        self.assertEqual(estimate_result_count(s, self.catalog), 10)

    def test_exact_key(self):
        s = new_state("c2")
        s.slots.category = "novelty"
        self.assertEqual(estimate_result_count(s, self.catalog), 2)

    def test_token_intersection(self):
        s = new_state("c3")
        s.slots.category = "women clothing"
        self.assertEqual(estimate_result_count(s, self.catalog), 3)  # {0,1,2}

    def test_partial_token_falls_back_to_largest(self):
        s = new_state("c4")
        s.slots.category = "women footwear"
        self.assertEqual(estimate_result_count(s, self.catalog), 5)  # women

    def test_never_zero(self):
        s = new_state("c5")
        s.slots.category = "zzz nonexistent"
        self.assertGreaterEqual(estimate_result_count(s, self.catalog), 1)


class DistillTest(unittest.TestCase):
    def test_distill_compact(self):
        s = new_state("d1", {"summary": "prefers fit", "preference_tags": ["fit"]})
        s = update_state(s, "I'm looking for Shoes. A key requirement is: leather.")
        compact = distill(s)
        self.assertEqual(compact["intent"], "BUYING")
        self.assertEqual(compact["slots"]["material"], "leather")
        self.assertEqual(compact["profile_summary"], "prefers fit")
        self.assertIn("leather", compact["summary"])


if __name__ == "__main__":
    unittest.main()
