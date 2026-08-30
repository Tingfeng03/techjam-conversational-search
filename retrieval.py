"""
retrieval.py — Person 1

Hybrid BM25 + dense vector retrieval with Reciprocal Rank Fusion (RRF).
Implements the retrieve() and rerank() stubs from interfaces.py.

Pipeline per query:
    1. BM25 keyword search          → top-200 ranked product indices
    2. Dense vector search          → top-200 ranked product indices
    3. RRF fusion (k=60)            → merged score list
    4. Hard filter application      → remove constraint violations
    5. Return top-K raw product dicts

Reranking (called after retrieve()):
    1. Build preference query from all accumulated slots + last user message
    2. Encode preference query with the same sentence-transformer
    3. Score each candidate: cosine_sim(preference_vec, product_vec)
       using pre-computed catalog embeddings — no extra encoding per candidate
    4. Small adjustments: gender (structural) + rating tiebreaker
    5. Return top-K sorted by score

To use from agent.py:
    from catalog import Catalog
    from retrieval import RetrievalPipeline

    catalog = Catalog("data/catalog.jsonl")        # created once at startup
    pipeline = RetrievalPipeline(catalog)           # lightweight wrapper

    # In respond():
    candidates = pipeline.retrieve(query, filters, top_k=50, buying_mode=True)
    ranked     = pipeline.rerank(candidates, state, top_k=10)
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

import bm25s
import numpy as np

from catalog import Catalog
from interfaces import ConversationState, Filters, Product

# LLM reranker — activated only when an API key is set. Mirrors the pattern in state.py.
_LLM_KEY   = os.environ.get("TECHJAM_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
_LLM_BASE  = os.environ.get("TECHJAM_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
_LLM_MODEL = os.environ.get("TECHJAM_LLM_MODEL", "gpt-4o-mini")

_LLM_SYSTEM = (
    "You are a shopping assistant reranker. Given a shopper's requirements and a list of products, "
    "rank the products by how well they match ALL requirements. "
    "Return ONLY a valid JSON array of parent_asin strings, best match first. "
    "Include exactly the number of items requested. No explanation, no extra text."
)


def _llm_rerank(slot_summary: str, candidates: list[dict], top_k: int) -> list[dict] | None:
    """Re-rank candidates using an LLM. Returns None on any failure (caller uses CE result)."""
    if not _LLM_KEY or not candidates:
        return None

    asin_to_product = {p.get("parent_asin", ""): p for p in candidates}

    product_list = []
    for p in candidates:
        price = Product._safe_price(p.get("price"))
        details = p.get("details") or {}
        detail_parts = []
        for key in ("Department", "Brand", "Material", "Color", "Size"):
            val = details.get(key)
            if val:
                detail_parts.append(f"{key}:{val}")
        product_list.append({
            "asin": p.get("parent_asin", ""),
            "title": (p.get("title") or "")[:80],
            "brand": p.get("store") or details.get("Brand") or "",
            "price": f"${price:.2f}" if price else "",
            "features": " | ".join((p.get("features") or [])[:3]),
            "details": " ".join(detail_parts),
        })

    user_msg = (
        f"Shopper requirements: {slot_summary}\n\n"
        f"Products:\n{json.dumps(product_list)}\n\n"
        f"Return the top {top_k} parent_asin values as a JSON array."
    )

    payload = {
        "model": _LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    }
    try:
        req = urllib.request.Request(
            f"{_LLM_BASE}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_LLM_KEY}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        # Accept either {"asins": [...]} or a bare list or any key whose value is a list
        if isinstance(parsed, list):
            asins = parsed
        elif isinstance(parsed, dict):
            asins = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            return None
        result = [asin_to_product[a] for a in asins if a in asin_to_product]
        return result[:top_k] if result else None
    except Exception:
        return None

_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange|"
    r"silver|gold|beige|navy|cream|burgundy|teal|maroon|ivory|coral|olive|tan)\b",
    re.I,
)
_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|"
    r"denim|suede|canvas|linen|fleece|satin|velvet|rubber|mesh|Gore-Tex|synthetic)\b",
    re.I,
)


def _searchable_text(product: dict) -> str:
    title = product.get("title") or ""
    store = product.get("store") or ""
    feats = " ".join((product.get("features") or [])[:5])
    cats = " ".join((product.get("categories") or [])[:3])
    desc = " ".join((product.get("description") or [])[:2])
    details = product.get("details") or {}
    dept = details.get("Department") or ""
    brand = details.get("Brand") or ""
    return f"{title} {store} {brand} {dept} {feats} {cats} {desc}".lower()


def _structural_score(product: dict, state: ConversationState) -> float:
    slots = state.slots
    text = _searchable_text(product)
    score = 0.0
    matched = 0

    if slots.brand:
        brand_low = slots.brand.lower()
        store_low = (product.get("store") or "").lower()
        details_brand = ((product.get("details") or {}).get("Brand") or "").lower()
        if brand_low in store_low or brand_low in details_brand:
            score += 0.25
            matched += 1

    if slots.color:
        if slots.color.lower() in text:
            score += 0.20
            matched += 1

    if slots.material:
        if slots.material.lower() in text:
            score += 0.20
            matched += 1

    if slots.gender:
        prod_gender = Product.from_dict(product).gender_from_details()
        if prod_gender == slots.gender.lower():
            score += 0.20
            matched += 1

    if slots.price_max is not None or slots.price_min is not None:
        safe_price = Product._safe_price(product.get("price"))
        if safe_price is not None:
            in_range = True
            if slots.price_max is not None and safe_price > slots.price_max:
                in_range = False
            if slots.price_min is not None and safe_price < slots.price_min:
                in_range = False
            if in_range:
                score += 0.15
                matched += 1

    if slots.use_case:
        if slots.use_case.lower() in text:
            score += 0.15
            matched += 1

    if slots.size:
        if slots.size.lower() in text:
            score += 0.10
            matched += 1

    for feat in (slots.features or []):
        if feat.lower() in text:
            score += 0.10
            matched += 1
        if matched >= 8:
            break

    return score

# 200 gives each arm enough breadth to catch products that rank poorly in one method
# but highly in the other. After RRF fusion the combined list is still cut to top_k (50),
# so this overhead only affects the merge step, not what gets returned to the caller.
_ARM_TOP_K = 200

# 60 is the standard RRF constant from the original Cormack et al. 2009 paper.
# Without it, 1/rank at rank 1 = 1.0 but at rank 2 = 0.5 — a huge cliff that makes
# the #1 result from either arm dominate regardless of the other arm's opinion.
# With k=60, rank 1 → 1/61 ≈ 0.016 and rank 2 → 1/62 ≈ 0.016 — much smoother,
# so a product that ranks #3 in both arms beats one that ranks #1 in only one arm.
_RRF_K = 60


class RetrievalPipeline:
    """
    Wraps a loaded Catalog to provide retrieve() and rerank().
    One instance shared across all sessions (stateless).
    """

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    # ------------------------------------------------------------------
    # Public API matching interfaces.py stubs
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        filters: Filters,
        top_k: int = 50,
        buying_mode: bool = False,
        category: str | None = None,
        hyde_query: str | None = None,
    ) -> list[dict]:
        """
        Hybrid BM25 + vector retrieval with RRF fusion and hard filtering.

        Args:
            query       : Free-text query built from slots
            filters     : Hard constraints (price, brand, rejected)
            top_k       : Number of candidates to return
            buying_mode : If True, hard-exclude filter violations;
                          if False, include more broadly (browsing)
            category    : If set, restrict search to products in this category

        Returns:
            List of up to top_k raw product dicts, best first.
        """
        if not query or not query.strip():
            query = "clothing shoes jewelry"

        allowed = self._get_category_indices(category) if category else None

        bm25_results = self._bm25_search(query, top_k=_ARM_TOP_K, allowed_indices=allowed)
        vec_results = self._vector_search(hyde_query or query, top_k=_ARM_TOP_K, allowed_indices=allowed)

        fused = self._rrf_merge(bm25_results, vec_results)

        filtered = self._apply_filters(fused, filters, strict=buying_mode)

        result = []
        for idx, _score in filtered[:top_k]:
            result.append(self.catalog.products[idx])
        return result

    def rerank(
        self,
        candidates: list[dict],
        state: ConversationState,
        top_k: int = 10,
    ) -> list[dict]:
        """Three-pass reranking: structural match → bi-encoder → cross-encoder."""
        if not candidates:
            return []

        rejected = set(state.rejected_asins or [])
        slots = state.slots

        parts: list[str] = []
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

        pref_tags = (state.user_profile or {}).get("preference_tags") or []
        parts.extend(pref_tags)

        pref_query = " ".join(parts).strip()

        if not pref_query:
            non_rejected = [p for p in candidates if p.get("parent_asin") not in rejected]
            return non_rejected[:top_k]

        q_vec = self.catalog.encoder.encode(
            pref_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        asin_to_idx = self.catalog.asin_to_idx
        gender = (slots.gender or "").lower()

        filled_slots = sum(1 for v in (
            slots.brand, slots.color, slots.material, slots.gender,
            slots.use_case, slots.size,
        ) if v) + (1 if slots.price_max or slots.price_min else 0) + min(len(slots.features or []), 2)

        if filled_slots >= 3:
            w_struct, w_sem, w_rating = 0.60, 0.30, 0.02
        elif filled_slots >= 1:
            w_struct, w_sem, w_rating = 0.40, 0.45, 0.02
        else:
            w_struct, w_sem, w_rating = 0.10, 0.70, 0.02

        scored: list[tuple[float, int, dict]] = []

        for rank, p in enumerate(candidates):
            asin = p.get("parent_asin", "")
            if asin in rejected:
                continue

            struct = _structural_score(p, state)

            idx = asin_to_idx.get(asin)
            sim = float(self.catalog.embeddings[idx] @ q_vec) if idx is not None else 0.0

            gender_adj = 0.0
            if gender:
                prod_gender = Product.from_dict(p).gender_from_details()
                if prod_gender == gender:
                    gender_adj = 0.10
                elif prod_gender and prod_gender != gender:
                    gender_adj = -0.15

            rating_bonus = ((p.get("average_rating") or 0.0) / 5.0) * w_rating

            score = w_struct * struct + w_sem * (sim + gender_adj) + rating_bonus
            scored.append((score, rank, p))

        scored.sort(key=lambda x: (-x[0], x[1]))
        all_scored = [p for _, _, p in scored]
        pass2 = all_scored[:30]

        ce_result = self._cross_encoder_rerank(pref_query, pass2, top_k=top_k)

        if not _LLM_KEY:
            return ce_result

        # Build slot summary for the LLM prompt
        slot_parts = []
        if slots.category:  slot_parts.append(f"category={slots.category}")
        if slots.brand:     slot_parts.append(f"brand={slots.brand}")
        if slots.gender:    slot_parts.append(f"gender={slots.gender}")
        if slots.use_case:  slot_parts.append(f"use_case={slots.use_case}")
        if slots.color:     slot_parts.append(f"color={slots.color}")
        if slots.material:  slot_parts.append(f"material={slots.material}")
        if slots.size:      slot_parts.append(f"size={slots.size}")
        if slots.price_max: slot_parts.append(f"price_max=${slots.price_max:.2f}")
        if slots.price_min: slot_parts.append(f"price_min=${slots.price_min:.2f}")
        if slots.features:  slot_parts.append(f"features={slots.features}")
        slot_summary = ", ".join(slot_parts) or pref_query

        # LLM sees the full top-50, not just the top-30 the CE saw
        llm_pool = all_scored[:50]
        llm_result = _llm_rerank(slot_summary, llm_pool, top_k=top_k)
        return llm_result if llm_result else ce_result

    def _cross_encoder_rerank(self, query: str, candidates: list[dict], top_k: int = 10) -> list[dict]:
        """Re-score candidates with the cross-encoder and return top_k sorted by score."""
        if not candidates:
            return candidates

        pairs = []
        for p in candidates:
            title = p.get("title") or ""
            store = p.get("store") or ""
            features = " ".join((p.get("features") or [])[:3])
            cats = [c for c in (p.get("categories") or [])
                    if c.lower() not in {"clothing, shoes & jewelry", "clothing shoes & jewelry"}]
            cat_text = " ".join(cats[:3])
            doc = " | ".join(filter(None, [title, store, features, cat_text]))
            pairs.append([query, doc])

        scores = self.catalog.cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [p for _, p in ranked[:top_k]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_category_indices(self, category: str) -> set[int] | None:
        """
        Return the set of product indices belonging to the given category.
        Tries the full string first, then each word individually.
        Skips overly broad words that would match most of the catalog.
        Returns None if no match or too broad (caller falls back to full catalog).
        """
        cat_index = self.catalog.category_index
        key = category.lower().strip()
        if not key:
            return None

        _SKIP = {"clothing", "shoes", "jewelry", "men", "women", "mens", "womens",
                 "boys", "girls", "men's", "women's", "shop", "accessories"}

        # Try full string first
        if key in cat_index:
            return set(cat_index[key])

        # Try each word — pick the most specific match (smallest result set > 0)
        best: set[int] | None = None
        for word in key.split():
            word = word.strip()
            if not word or word in _SKIP:
                continue
            matched: set[int] = set()
            for cat_key, indices in cat_index.items():
                if word in cat_key:
                    matched.update(indices)
            if matched and (best is None or len(matched) < len(best)):
                best = matched

        if best and len(best) < len(self.catalog.products) * 0.5:
            return best
        return None

    def _bm25_search(
        self, query: str, top_k: int, allowed_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return (product_idx, bm25_score) pairs for top_k BM25 results."""
        fetch_k = top_k if allowed_indices is None else min(top_k * 5, len(self.catalog.products))
        query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        results, scores = self.catalog.bm25.retrieve(query_tokens, k=min(fetch_k, len(self.catalog.products)), show_progress=False)
        out = []
        for idx, score in zip(results[0], scores[0]):
            if score <= 0:
                continue
            idx_int = int(idx)
            if allowed_indices is not None and idx_int not in allowed_indices:
                continue
            out.append((idx_int, float(score)))
            if len(out) >= top_k:
                break
        return out

    def _vector_search(
        self, query: str, top_k: int, allowed_indices: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        """Return (product_idx, cosine_sim) pairs for top_k vector results."""
        q_vec = self.catalog.encoder.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        sims = self.catalog.embeddings @ q_vec

        if allowed_indices is not None:
            mask = np.full(len(sims), -np.inf)
            for i in allowed_indices:
                mask[i] = sims[i]
            sims = mask

        k = min(top_k, int((sims > -np.inf).sum()))
        if k == 0:
            return []
        top_indices = np.argpartition(sims, -k)[-k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
        return [(int(idx), float(sims[idx])) for idx in top_indices if sims[idx] > -np.inf]

    def _rrf_merge(
        self,
        bm25_results: list[tuple[int, float]],
        vec_results: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Reciprocal Rank Fusion.
        score(d) = 1/(rank_bm25 + k) + 1/(rank_vec + k)
        Missing from one arm → treated as rank = len(results) + 1
        """
        k = _RRF_K
        rrf_scores: dict[int, float] = {}

        # RRF discards raw scores entirely and works only from rank position.
        # This sidesteps the incompatibility between BM25 scores (unbounded floats,
        # typically 0–20) and cosine similarity (always 0–1) — there is no meaningful
        # way to add or normalise those two scales against each other.
        for rank, (idx, _) in enumerate(bm25_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + k)

        for rank, (idx, _) in enumerate(vec_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (rank + k)

        # Products that appear in both arms receive two additive contributions,
        # naturally floating to the top — that's the hybrid benefit.
        merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return merged  # list[(idx, rrf_score)]

    # Tested and is worse than RRF fusion, but left here if we wanna try again
    def _dbsf_merge(
        self,
        bm25_results: list[tuple[int, float]],
        vec_results: list[tuple[int, float]],
    ) -> list[tuple[int, float]]:
        """
        Distribution-Based Score Fusion.
        Normalise each arm's raw scores to z-scores, then add.
        Preserves score gaps: a blowout BM25 hit stays dominant after fusion.
        """
        def z_normalise(results: list[tuple[int, float]]) -> list[tuple[int, float]]:
            if len(results) < 2:
                return [(idx, 0.0) for idx, _ in results]
            scores = np.array([s for _, s in results], dtype=np.float64)
            mean = scores.mean()
            std = scores.std()
            if std < 1e-9:
                return [(idx, 0.0) for idx, _ in results]
            return [(idx, float((s - mean) / std)) for idx, s in results]

        bm25_z = z_normalise(bm25_results)
        vec_z = z_normalise(vec_results)

        combined: dict[int, float] = {}
        for idx, z in bm25_z:
            combined[idx] = combined.get(idx, 0.0) + z
        for idx, z in vec_z:
            combined[idx] = combined.get(idx, 0.0) + z

        merged = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return merged

    def _apply_filters(
        self,
        ranked: list[tuple[int, float]],
        filters: Filters,
        strict: bool,
    ) -> list[tuple[int, float]]:
        """
        Apply hard constraints from Filters.

        strict=True  (BUYING mode): exclude any product violating a constraint.
        strict=False (BROWSING mode): only exclude rejected ASINs; let price/brand
                      violations through (reranker handles soft penalties).

        IMPORTANT:
          - price filter guards None: 78.9% of catalog has no price — never exclude these.
          - brand filter matches against product["store"] (NOT "brand" — that field doesn't exist).
        """
        rejected = set(filters.rejected_asins or [])
        products = self.catalog.products
        result = []

        for idx, score in ranked:
            p = products[idx]
            asin = str(p.get("parent_asin", ""))

            # Always exclude rejected
            if asin in rejected:
                continue

            if strict:
                # 78.9% of catalog products have no price field — excluding them when
                # the user sets a budget would wipe out most results. We only filter
                # products where a price is actually known and exceeds the limit.
                if filters.price_max is not None:
                    raw_price = p.get("price")
                    safe_price = Product._safe_price(raw_price)
                    if safe_price is not None and safe_price > filters.price_max:
                        continue

                if filters.price_min is not None:
                    raw_price = p.get("price")
                    safe_price = Product._safe_price(raw_price)
                    if safe_price is not None and safe_price < filters.price_min:
                        continue

                # The catalog has no top-level "brand" field — brand lives in "store"
                # (the seller/store name). Substring match handles cases like the user
                # typing "Nike" matching "Nike Official Store". details["Brand"] is a
                # secondary fallback used by ~2,328 products that populate it instead.
                if filters.brand:
                    brand_needle = filters.brand.lower()
                    store_haystack = (p.get("store") or "").lower()
                    details_brand = (
                        (p.get("details") or {}).get("Brand") or ""
                    ).lower()
                    if brand_needle not in store_haystack and brand_needle not in details_brand:
                        continue

            result.append((idx, score))

        return result


# ---------------------------------------------------------------------------
# Module-level convenience functions matching interfaces.py stub signatures
# ---------------------------------------------------------------------------

_pipeline: RetrievalPipeline | None = None


def _get_pipeline() -> RetrievalPipeline:
    """Lazy-load the pipeline using default catalog path."""
    global _pipeline
    if _pipeline is None:
        cat = Catalog()
        _pipeline = RetrievalPipeline(cat)
    return _pipeline


def retrieve(
    query: str,
    filters: Filters,
    top_k: int = 50,
    buying_mode: bool = False,
    category: str | None = None,
    hyde_query: str | None = None,
) -> list[dict]:
    """Drop-in replacement for the retrieve() stub in interfaces.py."""
    return _get_pipeline().retrieve(query, filters, top_k=top_k, buying_mode=buying_mode, category=category, hyde_query=hyde_query)


def rerank(
    candidates: list[dict],
    state: ConversationState,
    top_k: int = 10,
) -> list[dict]:
    """Drop-in replacement for the rerank() stub in interfaces.py."""
    return _get_pipeline().rerank(candidates, state, top_k=top_k)
