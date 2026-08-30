from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from catalog import Catalog
from interfaces import ConversationState
from retrieval import RetrievalPipeline
from state import new_state, update_state, estimate_result_count
from starter.orchestration.orchestration import (
    Orchestrator,
    build_filters,
    build_query,
    build_hyde_query,
    generate_clarification,
)
from starter.orchestration.responder import Responder


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog = Catalog(catalog_path)
        self.retrieval = RetrievalPipeline(self.catalog)
        self.orchestrator = Orchestrator()
        self.responder = Responder()
        self._sessions: set[str] = set()
        self._states: dict[str, ConversationState] = {}
        self.state = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions.add(session_id)
        self._states[session_id] = new_state(session_id, user_profile)
        self.state = self._states[session_id]

    def respond(self, session_id, user_message, turn, top_k) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        current_state = self._states[session_id]
        current_state.turn_count = turn
        state = update_state(current_state, user_message)
        state.turn_count = turn
        self._states[session_id] = state
        self.state = state

        est = estimate_result_count(state, self.catalog)
        decision = self.orchestrator.decide(state, est)

        if decision.action == "CLARIFY":
            previously_asked = set(state.asked_clarifications)
            question = generate_clarification(decision.missing_slots, state)
            if question:
                asked = next(
                    (slot for slot in decision.missing_slots
                     if slot not in previously_asked and slot in state.asked_clarifications),
                    decision.missing_slots[0] if decision.missing_slots else None,
                )
                return self.responder.format_response([], "CLARIFY", question, asked_slot=asked)
            decision.action = "SEARCH"

        if decision.action in ("SEARCH", "SEARCH_AND_CLARIFY"):
            query = build_query(state)
            hyde_query = build_hyde_query(state)
            filters = build_filters(state)
            candidates = self.retrieval.retrieve(
                query, filters, top_k=100,
                buying_mode=(state.intent == "BUYING"),
                category=state.slots.category,
                hyde_query=hyde_query,
            )
            ranked = self.retrieval.rerank(candidates, state, top_k=10)
            state.last_candidates = ranked
            self._states[session_id] = state

            if decision.action == "SEARCH_AND_CLARIFY":
                # Question selection happens after retrieval so it can use the
                # actual candidate set shown to the user.  This keeps
                # orchestration independent of the retrieval implementation
                # while making clarification genuinely adaptive.
                decision.missing_slots = self.orchestrator.rank_clarifications(
                    state, ranked,
                )
                previously_asked = set(state.asked_clarifications)
                question = generate_clarification(decision.missing_slots, state)
                if question:
                    decision.reason = self.orchestrator.clarification_diagnostic(
                        state, ranked, decision.missing_slots[0],
                    )
                    asked = next(
                        (slot for slot in decision.missing_slots
                         if slot not in previously_asked and slot in state.asked_clarifications),
                        decision.missing_slots[0] if decision.missing_slots else None,
                    )
                    return self.responder.format_response(
                        ranked, "SEARCH_AND_CLARIFY", question, asked_slot=asked,
                    )
            return self.responder.format_response(ranked, "SEARCH")

        return self.responder.format_response([], "FALLBACK")
