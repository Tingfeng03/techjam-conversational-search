from __future__ import annotations

import unittest

from domain_schema import AttributeSpec, DomainSchema
from interfaces import ConversationState
from starter.orchestration.orchestration import Orchestrator, generate_clarification


class DomainSchemaTest(unittest.TestCase):
    def test_validation_and_alias_resolution(self):
        schema = DomainSchema(
            domain_id="demo",
            default_query="demo",
            attributes=(
                AttributeSpec("topic", aliases=("subject",), required_for_search=True),
                AttributeSpec("duration", clarification_template="How long?"),
            ),
        )
        self.assertIs(schema.resolve("SUBJECT"), schema.get("topic"))
        self.assertEqual(schema.clarifiable()[0].name, "duration")

    def test_generic_orchestrator_does_not_require_clothing_names(self):
        schema = DomainSchema(
            domain_id="demo",
            default_query="demo",
            attributes=(
                AttributeSpec("topic", required_for_search=True),
                AttributeSpec("duration", clarification_template="How long?", product_values=lambda p: {p["duration"]} if p.get("duration") else set()),
            ),
        )
        state = ConversationState()
        state.slots.category = "demo"
        orchestrator = Orchestrator(schema=schema)
        self.assertEqual(orchestrator.rank_clarifications(state, [{"duration": "short"}, {"duration": "long"}]), ["duration"])
        self.assertEqual(generate_clarification(["duration"], state, schema), "How long?")

    def test_invalid_catch_all_fails_fast(self):
        with self.assertRaises(ValueError):
            DomainSchema("bad", "bad", (AttributeSpec("x"),), catch_all_attribute="missing")


if __name__ == "__main__":
    unittest.main()
