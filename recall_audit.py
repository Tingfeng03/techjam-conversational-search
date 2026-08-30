"""
recall_audit.py — For every miss session, identify at which pipeline stage
the target ASIN first disappears.

Stages checked per retrieve() call:
  1. BM25 arm top-200       — did BM25 find it?
  2. Vector arm top-200     — did vector search find it?
  3. RRF merged (all)       — did it survive fusion (before cutoff)?
  4. Post-filter top-50     — did it survive hard filters + top-50 cutoff?
  5. Post-rerank top-10     — did reranking keep it?

For each miss session, we record the "best stage" reached across ALL turns
that triggered a search (i.e. whether the target was EVER retrievable).

Usage:
    HF_HOME=/private/tmp/claude-501/hf_cache \
    /private/tmp/claude-501/tj_venv/bin/python3 recall_audit.py
"""

from __future__ import annotations

import json
import random
import sys
import os
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bm25s
from catalog import Catalog
from retrieval import RetrievalPipeline, _ARM_TOP_K, _RRF_K
from interfaces import ConversationState, Filters, Product
from starter.agent import Agent
from evaluator.local_evaluator import (
    MAX_TURNS, TOP_K,
    load_jsonl, catalog_index, coarse_category,
    initial_message, customer_reply, normalize_recommendations,
    materialize_hidden_fields,
)


# ---------------------------------------------------------------------------
# Instrumented pipeline
# ---------------------------------------------------------------------------

class AuditPipeline(RetrievalPipeline):
    """Records per-stage recall for each retrieve()/rerank() call."""

    def __init__(self, catalog: Catalog) -> None:
        super().__init__(catalog)
        self.current_target: str | None = None
        # Per-call records; reset between sessions
        self._records: list[dict] = []

    def reset_session(self, target: str) -> None:
        self.current_target = target
        self._records = []

    def retrieve(self, query, filters, top_k=50, buying_mode=False, category=None):
        target = self.current_target
        if not query or not query.strip():
            query = "clothing shoes jewelry"

        allowed = self._get_category_indices(category) if category else None

        bm25_results = self._bm25_search(query, top_k=_ARM_TOP_K, allowed_indices=allowed)
        vec_results  = self._vector_search(query, top_k=_ARM_TOP_K, allowed_indices=allowed)
        fused        = self._rrf_merge(bm25_results, vec_results)
        filtered     = self._apply_filters(fused, filters, strict=buying_mode)

        products = self.catalog.products

        bm25_asins = {int(idx) for idx, _ in bm25_results}
        vec_asins  = {int(idx) for idx, _ in vec_results}
        rrf_asins  = {int(idx) for idx, _ in fused}
        filt_asins = {int(idx) for idx, _ in filtered[:top_k]}

        # Map target ASIN → catalog index
        target_idx = self.catalog.asin_to_idx.get(target) if target else None

        record = {
            "query": query,
            "in_bm25":     target_idx in bm25_asins if target_idx is not None else False,
            "in_vec":      target_idx in vec_asins  if target_idx is not None else False,
            "in_rrf":      target_idx in rrf_asins  if target_idx is not None else False,
            "in_filtered": target_idx in filt_asins if target_idx is not None else False,
            "in_reranked": False,  # filled by rerank()
            "target_exists_in_catalog": target_idx is not None,
        }
        self._records.append(record)

        result = []
        for idx, _score in filtered[:top_k]:
            result.append(products[idx])
        return result

    def rerank(self, candidates, state, top_k=10):
        # Skip the expensive cross-encoder for the audit — we only need to know
        # whether the target survives the full rerank. Use bi-encoder pass only
        # (pass 1) which is fast dot-product, then check membership.
        target = self.current_target
        if not candidates:
            return []

        # Run the bi-encoder pass (cheap) to get the same candidate pool the
        # cross-encoder would see, then just take top_k.
        import numpy as np
        rejected = set(state.rejected_asins or [])
        slots = state.slots
        parts = []
        if slots.category:  parts.append(slots.category)
        if slots.brand:     parts.append(slots.brand)
        if slots.gender:    parts.append(slots.gender)
        if slots.use_case:  parts.append(slots.use_case)
        if slots.color:     parts.append(slots.color)
        if slots.material:  parts.append(slots.material)
        if slots.style:     parts.append(slots.style)
        if slots.size:      parts.append(slots.size)
        parts.extend(slots.features or [])
        if state.last_query:
            parts.append(state.last_query)
        pref_query = " ".join(parts).strip()
        if not pref_query:
            result = [p for p in candidates if p.get("parent_asin") not in rejected][:top_k]
        else:
            q_vec = self.catalog.encoder.encode(
                pref_query, normalize_embeddings=True, convert_to_numpy=True,
            ).astype(np.float32)
            asin_to_idx = self.catalog.asin_to_idx
            scored = []
            for rank, p in enumerate(candidates):
                asin = p.get("parent_asin", "")
                if asin in rejected:
                    continue
                idx = asin_to_idx.get(asin)
                sim = float(self.catalog.embeddings[idx] @ q_vec) if idx is not None else 0.0
                scored.append((sim, rank, p))
            scored.sort(key=lambda x: (-x[0], x[1]))
            result = [p for _, _, p in scored[:top_k]]

        if self._records and target:
            reranked_asins = {p.get("parent_asin") for p in result}
            self._records[-1]["in_reranked"] = target in reranked_asins
        return result


# ---------------------------------------------------------------------------
# Instrumented Agent
# ---------------------------------------------------------------------------

class AuditAgent(Agent):
    def __init__(self, catalog_path="data/catalog.jsonl"):
        from catalog import Catalog
        self.catalog  = Catalog(catalog_path)
        self.pipeline = AuditPipeline(self.catalog)
        from starter.orchestration.orchestration import Orchestrator
        self.orchestrator = Orchestrator()
        self._sessions: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Run evaluation with audit
# ---------------------------------------------------------------------------

def run_audit(
    catalog_path: str = "data/catalog.jsonl",
    dataset_path: str = "data/public_set.jsonl",
) -> None:
    samples = load_jsonl(dataset_path)
    catalog_ids, categories, products = catalog_index(catalog_path)
    agent = AuditAgent(catalog_path)
    pipe: AuditPipeline = agent.pipeline  # type: ignore[assignment]

    session_results = []

    for sample in samples:
        session_id = f"audit_{uuid.uuid4().hex}"
        target = str(sample["ground_truth"]["parent_asin"])
        agent.reset(session_id, sample["user_profile"])
        pipe.reset_session(target)

        effective_card, effective_behavior = materialize_hidden_fields(sample, products)
        effective_sample = {**sample, "intent_card": effective_card, "behavior": effective_behavior}

        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(
            effective_sample, coarse_category(categories.get(target, [])), disclosed
        )

        hit_turn: int | None = None
        best_rank: int | None = None
        all_records: list[dict] = []

        for turn in range(1, MAX_TURNS + 1):
            pipe._records = []
            try:
                response = agent.respond(session_id, user_message, turn, TOP_K)
            except Exception as exc:
                response = {"message": "", "ask_attribute": None, "recommendations": []}
            all_records.extend(pipe._records)

            if not isinstance(response, dict):
                response = {"message": "", "ask_attribute": None, "recommendations": []}

            ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                break
            if turn == MAX_TURNS:
                break

            override = effective_sample.get("behavior", {}).get("override") or {}
            if not override_applied and turn + 1 == int(override.get("turn", 3)):
                override_applied = True
                new_value = str(override.get("new_value", ""))
                if new_value:
                    disclosed.add(new_value)
                user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
            else:
                user_message, boundary_used = customer_reply(
                    effective_sample, response.get("ask_attribute"), disclosed, boundary_used
                )

        # Summarise: best stage the target ever reached across all turns
        def best_stage(records: list[dict]) -> str:
            if not records:
                return "no_search"
            # Check if target is even in catalog
            if not any(r["target_exists_in_catalog"] for r in records):
                return "not_in_catalog"
            if any(r["in_reranked"] for r in records):
                return "reranked_top10"
            if any(r["in_filtered"] for r in records):
                return "filtered_top50"
            if any(r["in_rrf"] for r in records):
                return "rrf_merged"
            if any(r["in_vec"] for r in records):
                return "vec_only"
            if any(r["in_bm25"] for r in records):
                return "bm25_only"
            return "missed_both_arms"

        session_results.append({
            "sample_id": sample["sample_id"],
            "scenario":  sample["scenario_type"],
            "hit":       hit_turn is not None,
            "hit_turn":  hit_turn,
            "best_rank": best_rank,
            "best_stage_reached": best_stage(all_records),
            "search_turns": len(all_records),
            "per_turn_stages": [
                {k: v for k, v in r.items() if k != "query"}
                for r in all_records
            ],
        })

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------
    total   = len(session_results)
    hits    = [s for s in session_results if s["hit"]]
    misses  = [s for s in session_results if not s["hit"]]

    print(f"\n{'='*60}")
    print(f"RECALL AUDIT  ({total} sessions, {len(hits)} hits, {len(misses)} misses)")
    print(f"{'='*60}")

    stage_order = [
        "reranked_top10",
        "filtered_top50",
        "rrf_merged",
        "vec_only",
        "bm25_only",
        "missed_both_arms",
        "not_in_catalog",
        "no_search",
    ]
    stage_labels = {
        "reranked_top10":  "In reranked top-10       (hit, should not be miss)",
        "filtered_top50":  "Survived filter, top-50 → DROPPED by reranker",
        "rrf_merged":      "In RRF list               → DROPPED by filter/top-50 cutoff",
        "vec_only":        "Vector arm only           → DROPPED at RRF merge (or BM25 buried it)",
        "bm25_only":       "BM25 arm only             → DROPPED at RRF merge (or vector buried it)",
        "missed_both_arms":"Missed BOTH arms          → retrieval doesn't find it at all",
        "not_in_catalog":  "Target not in catalog     → unanswerable",
        "no_search":       "No search triggered       → orchestrator never searched",
    }

    stage_counter = Counter(s["best_stage_reached"] for s in misses)
    print(f"\nMiss breakdown by pipeline stage (N={len(misses)}):")
    for stage in stage_order:
        count = stage_counter.get(stage, 0)
        if count == 0:
            continue
        pct = 100 * count / len(misses)
        label = stage_labels.get(stage, stage)
        print(f"  {count:4d} ({pct:5.1f}%)  {label}")

    print(f"\nBy scenario:")
    for scenario in ("buying", "browsing", "intent_override", "boundary"):
        sc_sessions = [s for s in session_results if s["scenario"] == scenario]
        sc_misses   = [s for s in sc_sessions if not s["hit"]]
        if not sc_sessions:
            continue
        hr = sum(s["hit"] for s in sc_sessions) / len(sc_sessions)
        ctr = Counter(s["best_stage_reached"] for s in sc_misses)
        top = ctr.most_common(2)
        top_str = ", ".join(f"{s}={n}" for s, n in top)
        print(f"  {scenario:16s}  HR={hr:.3f}  misses={len(sc_misses):3d}  top miss stages: {top_str}")

    # Detailed: misses where target was in top-50 but reranker dropped it
    rerank_drops = [s for s in misses if s["best_stage_reached"] == "filtered_top50"]
    if rerank_drops:
        print(f"\nReranker drops ({len(rerank_drops)} sessions) — target was in top-50 but fell out of top-10:")
        for s in rerank_drops[:10]:
            print(f"  {s['sample_id']:20s}  scenario={s['scenario']:16s}  searches={s['search_turns']}")

    both_arm_misses = [s for s in misses if s["best_stage_reached"] == "missed_both_arms"]
    if both_arm_misses:
        print(f"\nBoth-arm misses ({len(both_arm_misses)} sessions) — BM25 and vector both failed:")
        for s in both_arm_misses[:10]:
            print(f"  {s['sample_id']:20s}  scenario={s['scenario']:16s}  searches={s['search_turns']}")


if __name__ == "__main__":
    run_audit()
