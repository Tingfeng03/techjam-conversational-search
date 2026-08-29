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

import bm25s
import numpy as np

from catalog import Catalog
from interfaces import ConversationState, Filters, Product

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
        vec_results = self._vector_search(query, top_k=_ARM_TOP_K, allowed_indices=allowed)

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
        """
        Two-pass reranking pipeline:

        Pass 1 — bi-encoder dot product (fast, covers all candidates):
          Build a preference query from all accumulated slots, encode once,
          score each candidate via cosine similarity to its pre-computed vector.
          Gender structural field and rating tiebreaker applied here.
          Output: top_k candidates, best-first.

        Pass 2 — cross-encoder rerank (accurate, top_k candidates only):
          Feed (preference_query, product_text) pairs through a cross-encoder.
          The model reads query and document together — it can spot mismatches
          that a bi-encoder misses because bi-encoders encode independently.
          Output: same top_k candidates, better-ordered.

        Returns up to top_k products, best first.
        """
        if not candidates:
            return []

        rejected = set(state.rejected_asins or [])
        slots = state.slots

        # Build a preference query from every accumulated slot.
        # This is richer than the retrieval query, which only used the current turn's text.
        # By the time rerank() is called, the agent may have extracted brand, gender, color,
        # material etc. across multiple turns — concatenating them all gives the transformer
        # a full picture of what the user wants, not just what they said most recently.
        # last_query is appended too because it may contain signals (e.g. "cozy", "gift")
        # that haven't been parsed into a slot yet.
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

        # If no slots have been filled yet, fall back to retrieval order
        if not pref_query:
            non_rejected = [p for p in candidates if p.get("parent_asin") not in rejected]
            return non_rejected[:top_k]

        # Encode the preference query once and reuse it for all candidates.
        # The alternative — encoding each candidate individually — would cost one
        # transformer forward pass per candidate (50 calls vs 1). Product embeddings
        # are already pre-computed in catalog.embeddings, so we only ever need to
        # encode the query side at runtime.
        q_vec = self.catalog.encoder.encode(
            pref_query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        asin_to_idx = self.catalog.asin_to_idx
        gender = (slots.gender or "").lower()
        scored: list[tuple[float, int, dict]] = []

        for rank, p in enumerate(candidates):
            asin = p.get("parent_asin", "")
            if asin in rejected:
                continue

            # Both vectors are L2-normalised (done at build time in _compute_embeddings),
            # so dot product equals cosine similarity without needing the full formula.
            idx = asin_to_idx.get(asin)
            sim = float(self.catalog.embeddings[idx] @ q_vec) if idx is not None else 0.0

            # Gender is a binary structural attribute stored in details["Department"].
            # The embedding model understands gender semantically but is inconsistent —
            # a women's product might embed close to a men's query due to shared vocabulary.
            # Reading the explicit field is more reliable for this one attribute.
            # Penalty (-0.15) is larger than bonus (+0.10) because showing a wrong-gender
            # product is a worse experience than missing a right-gender one.
            gender_adj = 0.0
            if gender:
                prod_gender = Product.from_dict(p).gender_from_details()
                if prod_gender == gender:
                    gender_adj = 0.10
                elif prod_gender and prod_gender != gender:
                    gender_adj = -0.15

            # Cosine similarity scores cluster in ~0.3–0.7, so 0.02 max is genuinely
            # a tiebreaker — it only matters when two products are semantically identical.
            rating_bonus = ((p.get("average_rating") or 0.0) / 5.0) * 0.02

            score = sim + gender_adj + rating_bonus
            # rank (original retrieval position) is stored so that products with identical
            # floating-point scores sort deterministically rather than arbitrarily.
            scored.append((score, rank, p))

        scored.sort(key=lambda x: (-x[0], x[1]))
        pass1 = [p for _, _, p in scored[:top_k]]

        # Pass 2: cross-encoder rerank over the top_k from pass 1.
        # Cross-encoder reads (query, document) together — much more accurate than
        # dot product for distinguishing near-identical candidates in the final list.
        # We only run it on top_k (10) items, not the full 50, to keep latency low.
        return self._cross_encoder_rerank(pref_query, pass1)

    def _cross_encoder_rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Re-score candidates with the cross-encoder and return sorted by score."""
        if not candidates:
            return candidates

        # Build (query, product_text) pairs — cross-encoder needs both in one input.
        # We use the same concise text as the embedding (_embed_text equivalent):
        # title + store + top features + categories.
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
        return [p for _, p in ranked]

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
) -> list[dict]:
    """Drop-in replacement for the retrieve() stub in interfaces.py."""
    return _get_pipeline().retrieve(query, filters, top_k=top_k, buying_mode=buying_mode, category=category)


def rerank(
    candidates: list[dict],
    state: ConversationState,
    top_k: int = 10,
) -> list[dict]:
    """Drop-in replacement for the rerank() stub in interfaces.py."""
    return _get_pipeline().rerank(candidates, state, top_k=top_k)
