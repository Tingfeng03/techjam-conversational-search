"""
exploration/test_rerank.py

Tests rerank() by simulating what the agent knows at turn 1 and turn 3.

For each session we:
  1. retrieve() top-50 using a realistic turn-1 query
  2. rerank() with slots built from the evaluator's intent_card
  3. Compare: rank before rerank vs rank after rerank

Metrics reported:
  - Hit@10  : is the target in top-10 after rerank?
  - MRR     : 1/rank of target (0 if not in top-10)
  - Baseline: same metrics using raw retrieve() order (no rerank)

Run from techjam-conversational-search/ directory:
    HF_HOME=$TMPDIR/hf_cache $TMPDIR/tj_venv/bin/python3 exploration/test_rerank.py
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catalog import Catalog
from retrieval import RetrievalPipeline
from interfaces import Filters, ConversationState, Slots, Product

CATALOG_PATH = "data/catalog.jsonl"
SESSIONS_PATH = "data/public_set.jsonl"
RETRIEVE_K = 50
RERANK_K = 10
TEST_N = 50


# ---- replicate evaluator's intent_card logic --------------------------------

_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)


def _searchable_text(p):
    parts = []
    for field in ("title", "features", "details", "description", "categories", "store"):
        v = p.get(field)
        if isinstance(v, dict):
            parts.extend(f"{k} {item}" for k, item in v.items())
        elif isinstance(v, list):
            parts.extend(str(x) for x in v)
        elif v:
            parts.append(str(v))
    return " ".join(parts)


def _intent_card(p, limit=180):
    cands = []
    for v in (p.get("features") or []):
        cands.append(str(v))
    for k, v in (p.get("details") or {}).items():
        if v not in (None, "", []):
            cands.append(f"{k}: {v}")
    corpus = _searchable_text(p)
    m = _MATERIAL_RE.search(corpus)
    c = _COLOR_RE.search(corpus)
    if m:
        cands.insert(0, m.group(1).lower())
    if c:
        cands.insert(1, f"color: {c.group(1).lower()}")
    if p.get("price") not in (None, ""):
        cands.append(f"budget around ${p['price']}")
    cleaned = list(dict.fromkeys(x.strip()[:limit] for x in cands if x.strip()))
    return {
        "hard_constraints": cleaned[:2],
        "soft_preferences": cleaned[2:4] or cleaned[:1],
    }


def _coarse_category(cats):
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned = []
    for v in cats:
        for part in v.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def build_slots_from_card(card, category, scenario):
    """
    Simulate what Person 2's update_state() would extract after turn 1.
    We use the intent_card directly since we don't have a real NLP extractor yet.
    """
    slots = Slots()
    slots.category = category

    # Extract material from hard_constraints
    for c in card["hard_constraints"]:
        m = _MATERIAL_RE.search(c)
        if m:
            slots.material = m.group(1).lower()
        c_match = _COLOR_RE.search(c)
        if c_match:
            slots.color = c_match.group(1).lower()

    # Extract gender from category
    cat_lower = category.lower()
    if "women" in cat_lower or "girls" in cat_lower:
        slots.gender = "women"
    elif "men" in cat_lower or "boys" in cat_lower:
        slots.gender = "men"

    return slots


# ---- main -------------------------------------------------------------------

def mrr_at_k(rank, k=10):
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def main():
    print("=== Loading catalog ===\n")
    cat = Catalog(CATALOG_PATH)
    pipeline = RetrievalPipeline(cat)

    print("\n=== Loading sessions ===")
    with open(SESSIONS_PATH) as f:
        sessions = [json.loads(l) for l in f if l.strip()]
    sessions = sessions[:TEST_N]
    print(f"Testing {len(sessions)} sessions\n")

    results = []
    scenario_stats = {}

    header = f"{'#':<4} {'scenario':<15} {'target':<12} {'ret_rank':>8} {'rnk_rank':>8}  {'title snippet'}"
    print(header)
    print("-" * 88)

    for i, sess in enumerate(sessions):
        target = str(sess["ground_truth"]["parent_asin"])
        scenario = sess["scenario_type"]
        p = cat.products[cat.asin_to_idx[target]] if target in cat.asin_to_idx else {}

        # Build turn-1 query (same as honest test)
        cats = [str(x) for x in (p.get("categories") or [])]
        category = _coarse_category(cats)
        card = _intent_card(p)

        if scenario == "buying" and card["hard_constraints"]:
            query = f"{category} {card['hard_constraints'][0]}"
        elif scenario == "intent_override" and card["soft_preferences"]:
            query = f"{category} {card['soft_preferences'][-1]}"
        else:
            query = category

        # Retrieve top-50 with category pre-filtering
        candidates = pipeline.retrieve(query, Filters(), top_k=RETRIEVE_K, category=category)
        candidate_asins = [c["parent_asin"] for c in candidates]
        ret_rank = candidate_asins.index(target) + 1 if target in candidate_asins else None

        # Build ConversationState with slots extracted from intent_card
        slots = build_slots_from_card(card, category, scenario)
        state = ConversationState(session_id=f"test_{i}")
        state.slots = slots
        state.last_query = query
        state.user_profile = sess.get("user_profile", {})

        # Rerank
        ranked = pipeline.rerank(candidates, state, top_k=RERANK_K)
        ranked_asins = [r["parent_asin"] for r in ranked]
        rnk_rank = ranked_asins.index(target) + 1 if target in ranked_asins else None

        results.append({
            "scenario": scenario,
            "ret_rank": ret_rank,
            "rnk_rank": rnk_rank,
        })

        if scenario not in scenario_stats:
            scenario_stats[scenario] = {"n": 0, "ret_hit": 0, "rnk_hit": 0,
                                         "ret_mrr": 0.0, "rnk_mrr": 0.0}
        s = scenario_stats[scenario]
        s["n"] += 1
        if ret_rank and ret_rank <= RERANK_K:
            s["ret_hit"] += 1
        if rnk_rank:
            s["rnk_hit"] += 1
        s["ret_mrr"] += mrr_at_k(ret_rank)
        s["rnk_mrr"] += mrr_at_k(rnk_rank)

        ret_str = f"rank={ret_rank}" if ret_rank else "miss@50"
        rnk_str = f"rank={rnk_rank}" if rnk_rank else f"miss@{RERANK_K}"
        # Highlight improvements
        arrow = " ▲" if (rnk_rank and (ret_rank is None or rnk_rank < ret_rank)) else (
                " ▼" if (ret_rank and ret_rank <= RERANK_K and (rnk_rank is None or rnk_rank > ret_rank)) else "  ")
        title = (p.get("title") or "")[:35]
        print(f"{i+1:<4} {scenario:<15} {target:<12} {ret_str:>8} {rnk_str:>8}{arrow}  {title}")

    # Summary
    n = len(results)
    ret_hit_total = sum(1 for r in results if r["ret_rank"] and r["ret_rank"] <= RERANK_K)
    rnk_hit_total = sum(1 for r in results if r["rnk_rank"])
    ret_mrr_total = sum(mrr_at_k(r["ret_rank"]) for r in results) / n
    rnk_mrr_total = sum(mrr_at_k(r["rnk_rank"]) for r in results) / n

    print("-" * 88)
    print(f"\n{'':20} {'Baseline (retrieve order)':>26}   {'After rerank':>14}")
    print(f"{'OVERALL':20} Hit@{RERANK_K}: {ret_hit_total}/{n} = {ret_hit_total/n:.1%}   "
          f"MRR: {ret_mrr_total:.3f}   |   "
          f"Hit@{RERANK_K}: {rnk_hit_total}/{n} = {rnk_hit_total/n:.1%}   MRR: {rnk_mrr_total:.3f}")

    print("\nBy scenario:")
    for sc, s in sorted(scenario_stats.items()):
        r_h = s["ret_hit"] / s["n"]
        k_h = s["rnk_hit"] / s["n"]
        r_m = s["ret_mrr"] / s["n"]
        k_m = s["rnk_mrr"] / s["n"]
        print(f"  {sc:<15}  baseline Hit@{RERANK_K}={r_h:.1%} MRR={r_m:.3f}  →  "
              f"rerank Hit@{RERANK_K}={k_h:.1%} MRR={k_m:.3f}")

    delta_hit = rnk_hit_total - ret_hit_total
    delta_mrr = rnk_mrr_total - ret_mrr_total
    print(f"\nDelta:  Hit@{RERANK_K} {delta_hit:+d}   MRR {delta_mrr:+.3f}")
    if delta_mrr > 0:
        print("PASS — reranker improves MRR over raw retrieval order")
    else:
        print("WARN — reranker not improving MRR; check slot extraction quality")


if __name__ == "__main__":
    main()
