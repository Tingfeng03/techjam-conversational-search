from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from starter.orchestration.orchestration import (
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
    """Editable weak baseline: stateless BM25 retrieval with no LLM dependency."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog = Catalog(catalog_path)

        # TODO: load llm using config
        llm_client = 

        self.retrieval    = RetrievalPipeline(self.catalog)
        self.orchestrator = Orchestrator()
        self.reranker     = LLMReranker(llm_client)
        self.responder    = Responder(llm_client)
        self.llm          = llm_client

        self.state = None
        self.memory = None
        self.reset()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)
        self.state = ConversationState()
        self.memory = SessionMemory()

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")

        self.state.turn_count = turn  

        self.state = update_state(self.state, user_message)

        est = estimate_result_count(self.state, self.catalog)

        decision = decide(self.state, est)

        if decision.action == "CLARIFY":
            question = generate_clarification(decision.missing_slots, self.state)
            if question:
                asked = decision.missing_slots[0]
                return format_response([], "CLARIFY", question, asked_slot=asked)
            else:
                decision.action = "SEARCH"

        if decision.action == "SEARCH":
               query      = build_query(self.state)
               filters    = build_filters(self.state)
               candidates = retrieve(query, filters, top_k=50, buying_mode=(self.state.intent == "BUYING"))
               ranked     = rerank(candidates, self.state, top_k=10)    
               return format_response(ranked, "SEARCH")          


