"""
catalog.py — Person 1

Loads the 50k catalog once and builds:
  - BM25 index  (bm25s)
  - Dense embeddings  (multi-qa-MiniLM-L6-cos-v1, 384-dim, cached to .npy)
  - Cross-encoder reranker  (cross-encoder/ms-marco-MiniLM-L-6-v2)
  - category_index  {category_str: [product_idx, ...]} for estimate_result_count()

Usage:
    from catalog import Catalog
    cat = Catalog("data/catalog.jsonl")          # ~80 s first run, ~5 s after cache exists

    # If HuggingFace cache permission error, set env var before importing:
    #   export HF_HOME=$TMPDIR/hf_cache
    # or in Python before import: os.environ["HF_HOME"] = os.environ["TMPDIR"] + "/hf_cache"
    # cat.products    — list[dict], all 50k raw dicts
    # cat.bm25        — BM25Okapi instance
    # cat.embeddings  — np.ndarray shape (50000, 384)
    # cat.category_index — dict{str: list[int]}
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# Bi-encoder for catalog embeddings and query encoding.
# multi-qa-MiniLM-L6-cos-v1 scores 51.83 on MTEB retrieval vs ~49 for all-MiniLM-L6-v2
# and was trained on 215M QA pairs — closer to our search use case.
# Same 384-dim / 22M params / 750 queries/sec on CPU — direct drop-in.
EMBED_MODEL = "multi-qa-MiniLM-L6-cos-v1"

# Cross-encoder for the final rerank pass (top-10 only).
# Reads query + document together — fundamentally more accurate than dot product.
# 22M params, same MiniLM architecture, practical on CPU for 10 candidates.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _product_text(p: dict) -> str:
    """
    Build one searchable string per product for BM25 tokenisation.
    Weighted: title + store (brand) repeated for emphasis, then features, categories, description.
    """
    parts: list[str] = []

    title = p.get("title") or ""
    if title:
        parts.append(title)
        # BM25 scores by term frequency — repeating the title doubles its TF,
        # making title matches rank higher than the same word appearing in description.
        parts.append(title)

    store = p.get("store") or ""
    if store:
        parts.append(store)
        # Same doubling trick for brand — a brand-name query should strongly prefer
        # products from that brand over products that merely mention the brand in features.
        parts.append(store)

    features = p.get("features") or []
    if features:
        parts.append(" ".join(str(f) for f in features))

    cats = p.get("categories") or []
    # The root category "Clothing, Shoes & Jewelry" appears on every single product,
    # so including it adds zero discriminating power and inflates every product's score equally.
    relevant_cats = [c for c in cats if c.lower() not in
                     {"clothing, shoes & jewelry", "clothing shoes & jewelry", "clothing"}]
    if relevant_cats:
        parts.append(" ".join(relevant_cats))

    desc = p.get("description") or []
    if desc:
        # Later elements in the description list are usually repeated marketing boilerplate;
        # the first element contains the most product-specific text.
        first = str(desc[0]) if desc else ""
        if first:
            # BM25 builds an in-memory inverted index — very long descriptions multiply
            # memory usage across 50k products, so we cap to the most informative portion.
            parts.append(first[:300])

    details = p.get("details") or {}
    dept = details.get("Department") or ""
    if dept:
        parts.append(dept)
    mat = details.get("Material") or ""
    if mat:
        parts.append(mat)
    color = details.get("Color") or ""
    if color:
        parts.append(color)

    return " ".join(parts)


def _embed_text(p: dict) -> str:
    """
    Build the text fed to the sentence-transformer.
    Shorter than BM25 text — transformers have a 512-token limit.
    title + store + top-2 features + top-3 categories
    """
    parts: list[str] = []
    title = p.get("title") or ""
    if title:
        parts.append(title)
    store = p.get("store") or ""
    if store:
        parts.append(store)
    features = p.get("features") or []
    # Transformers have a 512 subword-token hard limit — text beyond that is silently
    # truncated. Feeding the full BM25 text (~500+ words) would lose the tail entirely.
    # Top-2 features capture the most differentiating product attributes without overflow.
    parts.extend(str(f) for f in features[:2])
    cats = p.get("categories") or []
    relevant = [c for c in cats if c.lower() not in
                {"clothing, shoes & jewelry", "clothing shoes & jewelry"}]
    # Top-3 non-root categories give the model enough hierarchy context (e.g.
    # "Women > Shoes > Running") without padding the input unnecessarily.
    parts.extend(relevant[:3])
    return " ".join(parts)


class Catalog:
    """
    Loaded once at startup. Shared across all sessions.

    Attributes:
        products       : list[dict] — all 50k raw catalog dicts, original order
        asin_to_idx    : dict[str, int] — fast ASIN → index lookup
        bm25           : bm25s.BM25 — keyword index over all products
        embeddings     : np.ndarray shape (N, 384) float32 — dense vectors, L2-normalised
        encoder        : SentenceTransformer — shared encoder (reused by retrieval.py)
        category_index : dict[str, list[int]] — lowercase category → product indices
                         used by estimate_result_count()
    """

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        embeddings_cache: str | Path = "catalog_embeddings.npy",
        batch_size: int = 512,
    ) -> None:
        catalog_path = Path(catalog_path)
        embeddings_cache = Path(embeddings_cache)

        print(f"[Catalog] Loading products from {catalog_path} …")
        self.products: list[dict] = []
        with catalog_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.products.append(json.loads(line))
        print(f"[Catalog] Loaded {len(self.products):,} products")

        # Reranking needs to look up the pre-computed embedding for each candidate product.
        # Iterating self.products to find a match would be O(N) per candidate — a dict
        # makes it O(1), which matters when reranking 50 candidates across every query.
        self.asin_to_idx: dict[str, int] = {
            str(p["parent_asin"]): i for i, p in enumerate(self.products)
        }

        # BM25 index (bm25s handles tokenisation internally)
        print("[Catalog] Building BM25 index …")
        corpus_texts = [_product_text(p) for p in self.products]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", show_progress=False)
        self.bm25 = bm25s.BM25()
        self.bm25.index(corpus_tokens, show_progress=False)
        print("[Catalog] BM25 index ready")

        # Category index (lowercase keys)
        print("[Catalog] Building category index …")
        self.category_index: dict[str, list[int]] = {}
        for idx, p in enumerate(self.products):
            for cat in (p.get("categories") or []):
                key = cat.lower().strip()
                if key:
                    if key not in self.category_index:
                        self.category_index[key] = []
                    self.category_index[key].append(idx)
        print(f"[Catalog] Category index has {len(self.category_index):,} entries")

        # Bi-encoder: used for catalog embedding and query encoding in retrieval
        print(f"[Catalog] Loading bi-encoder '{EMBED_MODEL}' …")
        self.encoder = SentenceTransformer(EMBED_MODEL)

        # Cross-encoder: used for the final rerank pass over top-10 candidates.
        # Loaded once here so retrieval.py can reference catalog.cross_encoder.
        print(f"[Catalog] Loading cross-encoder '{CROSS_ENCODER_MODEL}' …")
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        # Encoding 50k products through a transformer takes ~3 minutes — far too slow
        # to repeat on every startup. We save the result as a .npy binary (a raw float32
        # matrix) which loads in ~5 seconds via a single memory-mapped read.
        if embeddings_cache.exists():
            print(f"[Catalog] Loading embeddings from cache {embeddings_cache} …")
            self.embeddings: np.ndarray = np.load(str(embeddings_cache))
            # Row count mismatch means the catalog file was updated but the cache was not
            # deleted — recompute rather than silently serving stale embeddings.
            if self.embeddings.shape[0] != len(self.products):
                print("[Catalog] Cache size mismatch — recomputing …")
                self.embeddings = self._compute_embeddings(batch_size)
                np.save(str(embeddings_cache), self.embeddings)
        else:
            print("[Catalog] Computing embeddings (first run, ~3 min) …")
            self.embeddings = self._compute_embeddings(batch_size)
            np.save(str(embeddings_cache), self.embeddings)
            print(f"[Catalog] Embeddings cached to {embeddings_cache}")

        print(f"[Catalog] Ready — embeddings shape: {self.embeddings.shape}")

    def _compute_embeddings(self, batch_size: int) -> np.ndarray:
        """Encode all products and L2-normalise the vectors."""
        texts = [_embed_text(p) for p in self.products]
        vecs = self.encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            # L2 normalisation makes every vector length=1, so cosine similarity
            # reduces to a plain dot product: cos(a,b) = a·b when |a|=|b|=1.
            # This lets retrieval.py use fast matrix multiplication (embeddings @ q_vec)
            # instead of the slower full cosine formula at query time.
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # float32 halves memory vs float64 (50k × 384 × 4 bytes = ~77 MB vs ~154 MB)
        # with no meaningful precision loss for cosine similarity.
        return vecs.astype(np.float32)
