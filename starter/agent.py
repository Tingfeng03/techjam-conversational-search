from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

from catalog import Catalog
from retrieval import RetrievalPipeline
from state import new_state, update_state, estimate_result_count
from starter.orchestration.orchestration import (
    Orchestrator,
    build_filters,
    build_query,
    generate_clarification,
)
from starter.orchestration.responder import format_response


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog = Catalog(catalog_path)

        # The baseline is deliberately model-free.  A model client can be
        # injected by a future implementation without changing the protocol.
        llm_client = None

        self.retrieval    = RetrievalPipeline(self.catalog)
        self.orchestrator = Orchestrator()
        self.responder    = Responder(llm_client)
        self.llm          = llm_client

        self._sessions: set[str] = set()
        self._states: dict[str, ConversationState] = {}
        self.state = None
        self.memory = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)
        state = ConversationState(
            session_id=session_id,
            user_profile=dict(user_profile or {}),
        )
        self._states[session_id] = state
        # Keep these aliases for callers that inspect the most recently reset
        # session, while respond() always uses the requested session ID.
        self.state = state

    def respond(self, session_id, user_message, turn, top_k) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        # The real state implementation uses turn_count while parsing (notably
        # for intent overrides), so set it on the input state first.
        current_state = self._states[session_id]
        current_state.turn_count = turn
        state = update_state(current_state, user_message)
        # update_state may return a copied state, so apply evaluator metadata
        # after the update rather than relying on mutation.
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
            else:
                decision.action = "SEARCH"

        if decision.action in {"SEARCH", "SEARCH_AND_CLARIFY"}:
               query = build_query(state)
               filters = build_filters(state)
               limit = max(0, min(int(top_k), 10))
               candidates = self.retrieval.retrieve(
                   query, filters, top_k=50,
                   buying_mode=(state.intent == "BUYING"),
                   category=state.slots.category,
               )
               ranked = self.retrieval.rerank(candidates, state, top_k=limit)
               if decision.action == "SEARCH_AND_CLARIFY":
                   previously_asked = set(state.asked_clarifications)
                   question = generate_clarification(decision.missing_slots, state)
                   if question:
                       asked = next(
                           (slot for slot in decision.missing_slots
                            if slot not in previously_asked and slot in state.asked_clarifications),
                           decision.missing_slots[0] if decision.missing_slots else None,
                       )
                       return self.responder.format_response(
                           ranked, "SEARCH_AND_CLARIFY", question,
                           asked_slot=asked,
                       )
               return self.responder.format_response(ranked, "SEARCH")

        return self.responder.format_response([], "FALLBACK")

        return format_response([], "SEARCH")
