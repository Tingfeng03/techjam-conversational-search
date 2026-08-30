"""
exploration/test_retrieval.py

Honest retrieval test: queries are built the same way the evaluator does —
from coarse_category + first hard constraint/soft preference, NOT from the
product title. This reflects what the agent actually receives at turn 1.

Run from techjam-conversational-search/ directory:
    HF_HOME=$TMPDIR/hf_cache $TMPDIR/tj_venv/bin/python3 exploration/test_retrieval.py
"""
import json
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from catalog import Catalog
from retrieval import RetrievalPipeline
from interfaces import Filters

CATALOG_PATH = "data/catalog.jsonl"
SESSIONS_PATH = "data/public_set.jsonl"
TOP_K = 50
TEST_N = 50   # test first N sessions


# ---- replicate evaluator's query-construction logic -------------------------

_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)


def _searchable_text(p: dict) -> str:
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


def _intent_card(p: dict, limit: int = 180) -> dict:
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


def _coarse_category(cats: list) -> str:
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned = []
    for v in cats:
        for part in v.split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def build_turn1_query(p: dict) -> str:
    """
    Build the turn-1 query exactly as the evaluator constructs the initial user message.
    buying:         "{category}. {hard_constraint[0]}"
    browsing:       "{category}"
    intent_override:"{category}. {soft_preferences[-1]}"   (wrong direction first)
    boundary:       "{category}"
    We use the query text the agent would receive and must respond to.
    """
    card = _intent_card(p)
    cats = [str(x) for x in (p.get("categories") or [])]
    category = _coarse_category(cats)
    return category  # base: category is always present


def build_turn1_query_with_constraint(p: dict, scenario: str) -> str:
    """Include the first disclosed constraint in the query (buying/override turn 1)."""
    card = _intent_card(p)
    cats = [str(x) for x in (p.get("categories") or [])]
    category = _coarse_category(cats)

    if scenario == "buying" and card["hard_constraints"]:
        constraint = card["hard_constraints"][0]
        return f"{category} {constraint}"
    elif scenario == "intent_override" and card["soft_preferences"]:
        # turn 1 sends the OLD preference (wrong direction) — harder for retrieval
        old_value = card["soft_preferences"][-1]
        return f"{category} {old_value}"
    else:
        return category


# ---- main -------------------------------------------------------------------

def main():
    print("=== Loading catalog ===\n")
    cat = Catalog(CATALOG_PATH)
    pipeline = RetrievalPipeline(cat)

    print("\n=== Loading sessions ===")
    with open(SESSIONS_PATH) as f:
        sessions = [json.loads(l) for l in f if l.strip()]
    sessions = sessions[:TEST_N]
    print(f"Testing {len(sessions)} sessions\n")

    hits_cat_only = 0
    hits_with_constraint = 0
    scenario_stats: dict = {}

    print(f"{'#':<4} {'scenario':<15} {'target':<12} {'cat_only':>8} {'w/constraint':>13}  title snippet")
    print("-" * 90)

    for i, sess in enumerate(sessions):
        target = str(sess["ground_truth"]["parent_asin"])
        scenario = sess["scenario_type"]
        p = cat.products[cat.asin_to_idx[target]] if target in cat.asin_to_idx else {}

        # Query 1: category only (vaguer — browsing + boundary case)
        q_cat = build_turn1_query(p, ) if False else _coarse_category(
            [str(x) for x in (p.get("categories") or [])]
        )
        cands_cat = pipeline.retrieve(q_cat, Filters(), top_k=TOP_K)
        asins_cat = [x["parent_asin"] for x in cands_cat]
        rank_cat = asins_cat.index(target) + 1 if target in asins_cat else None

        # Query 2: category + first constraint (buying/override initial message)
        q_full = build_turn1_query_with_constraint(p, scenario)
        cands_full = pipeline.retrieve(q_full, Filters(), top_k=TOP_K)
        asins_full = [x["parent_asin"] for x in cands_full]
        rank_full = asins_full.index(target) + 1 if target in asins_full else None

        if rank_cat:
            hits_cat_only += 1
        if rank_full:
            hits_with_constraint += 1

        if scenario not in scenario_stats:
            scenario_stats[scenario] = {"cat": 0, "full": 0, "n": 0}
        scenario_stats[scenario]["n"] += 1
        if rank_cat:
            scenario_stats[scenario]["cat"] += 1
        if rank_full:
            scenario_stats[scenario]["full"] += 1

        cat_str = f"rank={rank_cat}" if rank_cat else "MISS"
        full_str = f"rank={rank_full}" if rank_full else "MISS"
        title = (p.get("title") or "")[:35]
        print(f"{i+1:<4} {scenario:<15} {target:<12} {cat_str:>8} {full_str:>13}  {title}")

    n = len(sessions)
    print("-" * 90)
    print(f"\n{'OVERALL':20} cat-only: {hits_cat_only}/{n} = {hits_cat_only/n:.1%}   "
          f"w/constraint: {hits_with_constraint}/{n} = {hits_with_constraint/n:.1%}")
    print()
    print("By scenario:")
    for sc, st in sorted(scenario_stats.items()):
        print(f"  {sc:<15}  cat-only: {st['cat']}/{st['n']} = {st['cat']/st['n']:.1%}   "
              f"w/constraint: {st['full']}/{st['n']} = {st['full']/st['n']:.1%}")

    print()
    if hits_with_constraint / n >= 0.70:
        print("PASS — retrieval covers targets well enough for reranker")
    else:
        print("WARN — hit rate below 70%, retrieval needs improvement")


if __name__ == "__main__":
    main()
