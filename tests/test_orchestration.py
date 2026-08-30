from __future__ import annotations

import unittest

from starter.orchestration.orchestration import Orchestrator, generate_clarification
from state import new_state


def product(index: int, **kwargs) -> dict:
    value = {
        "parent_asin": f"P{index}",
        "title": "shoe",
        "features": [],
        "categories": ["Shoes"],
        "details": {},
        "price": None,
        "store": "",
    }
    value.update(kwargs)
    return value


class OrchestrationPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = Orchestrator()
        self.state = new_state("test", {"preference_tags": []})
        self.state.slots.category = "shoes"
        self.state.turn_count = 1

    def test_candidate_diversity_changes_question_order(self) -> None:
        same_features = [product(i, features=["comfortable"]) for i in range(10)]
        varying_materials = [product(i, details={"Material": "cotton" if i % 2 else "leather"}) for i in range(10)]
        self.assertEqual(self.orchestrator.rank_clarifications(self.state, same_features)[0], "features")
        self.assertEqual(self.orchestrator.rank_clarifications(self.state, varying_materials)[0], "material")

    def test_known_attribute_is_discounted(self) -> None:
        candidates = [
            product(i, details={"Material": "cotton" if i % 2 else "leather", "Color": "blue" if i % 2 else "black"})
            for i in range(10)
        ]
        self.state.slots.material = "cotton"
        self.assertEqual(self.orchestrator.rank_clarifications(self.state, candidates)[0], "color")

    def test_asked_and_declined_attributes_are_skipped(self) -> None:
        candidates = [product(i, features=[f"feature-{i}"], details={"Material": "cotton"}) for i in range(10)]
        self.state.asked_clarifications.add("features")
        self.assertNotEqual(self.orchestrator.rank_clarifications(self.state, candidates)[0], "features")

    def test_profile_is_only_a_small_tiebreaker(self) -> None:
        candidates = [product(i, details={"Material": "cotton" if i % 2 else "leather"}) for i in range(10)]
        self.state.user_profile = {"preference_tags": ["fit"]}
        self.assertEqual(self.orchestrator.rank_clarifications(self.state, candidates)[0], "material")

    def test_late_other_recovery(self) -> None:
        self.state.turn_count = 6
        sparse = [product(i, categories=[], details={}) for i in range(10)]
        self.assertEqual(self.orchestrator.rank_clarifications(self.state, sparse), ["other"])

    def test_action_policy_searches_and_clarifies_until_last_turn(self) -> None:
        decision = self.orchestrator.decide(self.state, 1000)
        self.assertEqual(decision.action, "SEARCH_AND_CLARIFY")
        self.state.turn_count = 10
        decision = self.orchestrator.decide(self.state, 1000)
        self.assertEqual(decision.action, "SEARCH")

    def test_missing_category_is_the_only_pure_clarification(self) -> None:
        self.state.slots.category = None
        decision = self.orchestrator.decide(self.state, 1)
        self.assertEqual(decision.action, "CLARIFY")
        self.assertEqual(decision.missing_slots, ["category"])

    def test_public_question_mapping_for_features(self) -> None:
        question = generate_clarification(["features"], self.state)
        self.assertIsNotNone(question)
        self.assertIn("feature", question.lower())


if __name__ == "__main__":
    unittest.main()
