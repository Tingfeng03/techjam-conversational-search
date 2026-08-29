# Shopping Copilot — Detailed Implementation Plan

## Quick Reference: Evaluation Metrics

| Metric | What it means | How to improve |
|--------|--------------|----------------|
| **Hit Rate@K** | Is the target product in your top-K results? | Better retrieval |
| **MRR** | 1 / rank of the first correct result (higher = better) | Better reranking |
| **MTTC** | Mean turns before user gets the right product | Fewer, smarter clarifications |
| **TechnicalScore** | Combined formula using all three above | Balance all three |

Hard constraints: **max 10 turns per session**. Going over → score zero.

---

## Project File Structure

```
shopping_copilot/
├── agent.py          # Entry point — the class the evaluator calls
├── catalog.py        # Loads + indexes all 50k products
├── intent.py         # Classifies intent + extracts slots from user messages
├── state.py          # State machine + slot accumulation logic
├── retrieval.py      # BM25, vector, and hybrid search
├── orchestrator.py   # Decides: SEARCH vs CLARIFY vs RECOMMEND
├── reranker.py       # Reranks top-50 candidates using LLM or heuristics
├── responder.py      # Formats final response to the user
├── memory.py         # Short-term session memory + optional user profile
└── config.py         # Thresholds, constants, model names
```

---

## Component 1: Catalog (`catalog.py`)

**Purpose**: Load the 50k product JSONL, build BM25 index, compute vector embeddings — all in memory at startup.

### What data exists per product

**VERIFIED against actual data/catalog.jsonl** — use these exact field names or you'll get KeyErrors:
```
parent_asin    — unique product ID (e.g. "B07K34RX5J")  ← NOT "asin"
title          — product name string
description    — LIST of strings (NOT a single string) e.g. ["Long text..."]
                 → join with " ".join(p["description"]) to get searchable text
features       — list of strings e.g. ["Spandex", "Made in USA", "Lightweight..."]
price          — float e.g. 27.99, or null (~20% of products have no price)
store          — brand/store name e.g. "Spirit Hoops"  ← NOT "brand"
categories     — FLAT list of strings e.g. ["Clothing, Shoes & Jewelry", "Women", "Jewelry"]
                 → NOT a list of lists — it's already flat
details        — dict e.g. {"Department": "Womens", "Product Dimensions": "..."}
average_rating — float 0.0–5.0, or None
rating_number  — int review count, or None
```

**Fields that do NOT exist** (will cause KeyError — do not use):
- `"asin"` → use `"parent_asin"`
- `"brand"` → use `"store"`
- `"rating"` → use `"average_rating"`
- `"rating_count"` → use `"rating_number"`
- `"images"` → not in this dataset

### How to build it

```python
import json, re, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

class Catalog:
    def __init__(self, path: str, embeddings_cache: str = "catalog_embeddings.npy"):
        # Step 1: load raw products
        self.products = []          # list of dicts (one per product)
        self.asin_to_idx = {}       # asin -> integer index for fast lookup

        with open(path) as f:
            for line in f:
                p = json.loads(line)
                idx = len(self.products)
                self.products.append(p)
                self.asin_to_idx[p["parent_asin"]] = idx  # key is "parent_asin" not "asin"

        # Step 2: build BM25 index
        # Concatenate searchable text fields
        corpus_texts = [self._product_text(p) for p in self.products]
        tokenized = [self._tokenize(t) for t in corpus_texts]
        self.bm25 = BM25Okapi(tokenized)

        # Step 3: build or load vector embeddings
        # Uses ~80MB model, takes ~3 min to encode 50k products once
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        import os
        if os.path.exists(embeddings_cache):
            self.embeddings = np.load(embeddings_cache)   # shape (50000, 384)
        else:
            self.embeddings = self.encoder.encode(
                corpus_texts, batch_size=256, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True
            )
            np.save(embeddings_cache, self.embeddings)

        # Step 4: build category index for quick candidate count estimation
        self.category_index = {}   # "running shoes" -> [idx1, idx2, ...]
        for idx, p in enumerate(self.products):
            cats = self._flatten_categories(p.get("categories", []))
            for c in cats:
                self.category_index.setdefault(c.lower(), []).append(idx)

    def _product_text(self, p: dict) -> str:
        # "store" is the brand field. "description" is a list of strings.
        desc = p.get("description", [])
        desc_text = " ".join(desc) if isinstance(desc, list) else str(desc)
        parts = [p.get("title", ""), p.get("store", "")]
        parts += p.get("features", [])
        parts.append(desc_text)
        return " ".join(filter(None, parts))

    def _tokenize(self, text: str) -> list[str]:
        return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()

    def _flatten_categories(self, categories) -> list[str]:
        # categories is already a FLAT list of strings — no need to unwrap
        # e.g. ["Clothing, Shoes & Jewelry", "Women", "Jewelry", "Earrings"]
        return [c for c in categories if isinstance(c, str)]
```

**Why cache embeddings?** Encoding 50k products takes ~3 minutes on CPU. Caching them to `.npy` means startup is <5 seconds on subsequent runs. For the hackathon this is critical — you'll restart many times.

---

## Component 2: Intent & Slot Extractor (`intent.py`)

**Purpose**: Given a user message, classify the **intent** (BUYING/BROWSING) and extract **slots** (structured shopping constraints).

### Intent Classification

Two intents:
- **BUYING**: User has a specific target, wants precision. Hard constraints matter. ("I need Nike running shoes under $80")
- **BROWSING**: User is exploring, wants variety. ("Show me what summer dresses you have")

Use **embedding similarity** against a small set of labeled examples — semantic like an LLM, but free and instant. Reuses the same encoder already loaded for retrieval.

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# A handful of representative examples per intent.
# To tune: add misclassified messages from the 200 sessions to the correct bucket.
INTENT_EXAMPLES = {
    "BUYING": [
        "I need Nike running shoes under $80",
        "find me a red leather jacket",
        "I want to buy waterproof hiking boots size 10",
        "looking for a formal dress for a wedding",
        "get me something under $50",
    ],
    "BROWSING": [
        "show me what summer dresses you have",
        "what kind of sneakers do you carry",
        "I'm just browsing, what's popular",
        "any recommendations for winter coats",
        "what do you have in jewelry",
    ],
}

class IntentClassifier:
    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        # Pre-encode examples once at startup — negligible cost
        self._anchors = {}
        for intent, examples in INTENT_EXAMPLES.items():
            vecs = encoder.encode(examples, normalize_embeddings=True)
            self._anchors[intent] = vecs  # shape (N, 384)

    def classify(self, message: str) -> str:
        msg_vec = self.encoder.encode(message, normalize_embeddings=True)  # (384,)
        scores = {}
        for intent, anchors in self._anchors.items():
            # Max similarity to any anchor — one strong match is enough
            scores[intent] = float(np.max(anchors @ msg_vec))
        return max(scores, key=scores.get)
```

**Why this beats both alternatives:**

| | Keyword rules | LLM call | Embedding similarity |
|---|---|---|---|
| Handles paraphrases | No | Yes | Yes |
| Token cost | None | ~50 tokens/call | None |
| Latency | ~0ms | ~500ms | ~2ms |
| Reuses existing model | — | No | **Yes** |
| Tunable | Add more keywords | Change prompt | Add more examples |

### Slot Schema

These are the constraints we extract from user messages:

```python
# What we track in each session
SLOT_KEYS = {
    "category"   : str,    # "running shoes", "women's tops", "hoop earrings"
    "brand"      : str,    # "Nike", "Levi's"
    "price_min"  : float,  # 0.0
    "price_max"  : float,  # 100.0
    "color"      : str,    # "blue", "black"
    "size"       : str,    # "M", "10", "8.5"
    "gender"     : str,    # "men", "women", "unisex", "boys", "girls"
    "use_case"   : str,    # "running", "hiking", "formal", "casual"
    "features"   : list,   # ["waterproof", "lightweight", "breathable"]
    "style"      : str,    # "athletic", "formal", "bohemian"
    "material"   : str,    # "leather", "cotton", "synthetic"
}
```

### Slot Extraction with LLM

```python
SLOT_EXTRACTION_PROMPT = """You are extracting shopping constraints from a user message.

Current known constraints: {current_slots}
User just said: "{message}"

Extract ONLY what the user explicitly mentioned. Do NOT infer unstated values.
Return a JSON object. Use null for unknown fields.

Rules:
- "under $100" → price_max: 100, price_min: null
- "around $50" → price_min: 40, price_max: 60
- "between $30-80" → price_min: 30, price_max: 80
- "Actually, make it Adidas" → this overrides brand (set brand: "Adidas")
- If user says "forget Nike, I want Adidas", note the override

Return JSON with keys: category, brand, price_min, price_max, color, size,
gender, use_case, features (list), style, material, is_override (bool)"""

def extract_slots(message: str, state, llm) -> dict:
    prompt = SLOT_EXTRACTION_PROMPT.format(
        current_slots=json.dumps(state.slots, indent=2),
        message=message
    )
    raw = llm.complete(prompt, max_tokens=200)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: regex-based extraction
        return extract_slots_regex(message)
```

**Fallback regex extraction** (no LLM dependency):
```python
import re

def extract_slots_regex(message: str) -> dict:
    slots = {}
    msg = message.lower()

    # Price extraction
    price_pattern = re.search(r"under\s+\$?(\d+)", msg)
    if price_pattern:
        slots["price_max"] = float(price_pattern.group(1))

    between_pattern = re.search(r"between\s+\$?(\d+)\s*(?:and|-)\s*\$?(\d+)", msg)
    if between_pattern:
        slots["price_min"] = float(between_pattern.group(1))
        slots["price_max"] = float(between_pattern.group(2))

    # Gender detection
    if any(w in msg for w in ["women", "woman", "female", "ladies", "girl"]):
        slots["gender"] = "women"
    elif any(w in msg for w in ["men", "man", "male", "guys", "boy"]):
        slots["gender"] = "men"

    # Common brands in clothing/shoes/jewelry
    known_brands = ["nike", "adidas", "levi's", "h&m", "zara", "forever 21",
                    "under armour", "new balance", "skechers", "vans", "converse"]
    for brand in known_brands:
        if brand in msg:
            slots["brand"] = brand.title()
            break

    return slots
```

---

## Component 3: State Manager (`state.py`)

**Purpose**: Track the full conversational context — accumulated slots, intent, turn count, clarification history.

### The State Object

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ConversationState:
    # Core session info
    session_id: str = ""
    turn_count: int = 0            # increments each time user sends a message
    max_turns: int = 10            # hard limit from competition rules

    # What the user wants
    intent: str = "UNKNOWN"        # "BUYING" | "BROWSING" | "UNKNOWN"
    slots: dict = field(default_factory=dict)  # all extracted constraints

    # Conversation history (for LLM context)
    history: list = field(default_factory=list)
    # Each entry: {"role": "user"|"assistant", "content": "..."}

    # Search state
    last_candidates: list = field(default_factory=list)
    last_query: str = ""

    # Clarification tracking (avoid repeating questions)
    asked_clarifications: set = field(default_factory=set)

    # Memory
    rejected_asins: list = field(default_factory=list)  # products user said no to
```

### State Update Logic

This is the most important part — how new information from each turn gets merged.

Use an **OverrideDetector** (same embedding approach as `IntentClassifier`) — shares the already-loaded encoder, so no extra cost.

```python
import numpy as np

OVERRIDE_EXAMPLES = [
    "actually, make it Adidas instead",
    "forget Nike, I want Adidas",
    "no wait, change it to boots",
    "never mind the shoes, show me jackets",
    "actually I changed my mind",
    "let's start over, I want something different",
    "scratch that, I need a dress instead",
    "forget what I said, show me handbags",
]

NON_OVERRIDE_EXAMPLES = [
    "also make it blue",
    "and under $50",
    "preferably waterproof",
    "with good reviews",
    "size medium please",
]

class OverrideDetector:
    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder
        self._override_anchors     = encoder.encode(OVERRIDE_EXAMPLES,     normalize_embeddings=True)
        self._non_override_anchors = encoder.encode(NON_OVERRIDE_EXAMPLES, normalize_embeddings=True)

    def is_override(self, message: str) -> bool:
        msg_vec = self.encoder.encode(message, normalize_embeddings=True)
        override_score     = float(np.max(self._override_anchors     @ msg_vec))
        non_override_score = float(np.max(self._non_override_anchors @ msg_vec))
        return override_score > non_override_score
```

Both `IntentClassifier` and `OverrideDetector` are instantiated once at agent startup and passed in — they share the same encoder instance:

```python
encoder         = catalog.encoder   # already loaded
intent_clf      = IntentClassifier(encoder)
override_det    = OverrideDetector(encoder)
```

Then `update_state` uses the detector instead of the old hardcoded list:

```python
def update_state(
    state: ConversationState,
    new_slots: dict,
    new_intent: str,
    override_detector: OverrideDetector,
) -> ConversationState:
    """
    Merges new information into the state.
    Handles two scenarios:
    1. Information Accumulation: user adds more constraints ("also make it blue")
    2. Intent Override: user completely changes request ("actually, forget shoes, show me jackets")
    """
    last_user_msg = next(
        (m["content"] for m in reversed(state.history) if m["role"] == "user"), ""
    )
    is_override = new_slots.pop("is_override", False) or \
                  override_detector.is_override(last_user_msg)

    # Handle intent override
    if is_override:
        # User started over — clear old slots that conflict
        # Keep non-conflicting slots (e.g., budget usually carries over)
        carry_over = {k: v for k, v in state.slots.items() if k == "price_max"}
        state.slots = carry_over
        if new_intent and new_intent != "UNKNOWN":
            state.intent = new_intent

    # Update intent if we got a clearer signal
    elif new_intent and new_intent != "UNKNOWN":
        state.intent = new_intent

    # Merge new slots (new values override old ones for same key)
    for key, value in new_slots.items():
        if value is not None:
            if key == "features" and isinstance(value, list):
                # Accumulate features (don't replace)
                existing = state.slots.get("features", [])
                state.slots["features"] = list(set(existing + value))
            else:
                state.slots[key] = value

    return state
```

To tune after running the 200 sessions: add misclassified messages to `OVERRIDE_EXAMPLES` or `NON_OVERRIDE_EXAMPLES` — no code changes needed.

### Over-Generality Check

Before deciding to search, check if we'd get too many results:

```python
def estimate_result_count(state: ConversationState, catalog: Catalog) -> int:
    """Quick estimate of how many products would match current slots."""
    cat = state.slots.get("category", "").lower()
    if not cat:
        return len(catalog.products)   # 50k — way too many

    # Find products in this category
    matches = len(catalog.category_index.get(cat, []))
    if matches == 0:
        # Try fuzzy: check if any category key contains our word
        matches = sum(
            len(v) for k, v in catalog.category_index.items()
            if cat in k
        )
    return max(matches, 1)  # avoid returning 0


def is_over_general(state: ConversationState, catalog: Catalog) -> bool:
    estimated = estimate_result_count(state, catalog)
    # If more than 500 products would match AND we have few constraints
    meaningful_slots = [k for k in ["category", "brand", "use_case", "price_max"]
                        if state.slots.get(k)]
    return estimated > 500 and len(meaningful_slots) < 2
```

---

## Component 4: Retrieval Pipeline (`retrieval.py`)

**Purpose**: Given a query + filters, return the top-50 candidate products by combining BM25 keyword search and dense vector search.

### The Three Search Methods

#### Method A: BM25 (Keyword Search)

Good for exact keyword matches — "Nike Air Max" will score highly if those exact words appear.

```python
def bm25_search(self, query: str, top_k: int = 200) -> list[tuple[int, float]]:
    """Returns list of (product_index, score) sorted descending."""
    tokens = self._tokenize(query)
    scores = self.catalog.bm25.get_scores(tokens)
    # argsort in descending order
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
```

#### Method B: Vector Search (Semantic Search)

Good for semantic matches — "breathable footwear for jogging" finds running shoes even without those exact words.

```python
def vector_search(self, query: str, top_k: int = 200) -> list[tuple[int, float]]:
    """Returns list of (product_index, cosine_similarity) sorted descending."""
    # Encode query to same 384-dim space as product embeddings
    query_vec = self.catalog.encoder.encode(query, normalize_embeddings=True)
    # Embeddings are already L2-normalized, so dot product = cosine similarity
    similarities = np.dot(self.catalog.embeddings, query_vec)  # shape (50000,)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(i), float(similarities[i])) for i in top_indices]
```

#### Method C: Metadata Filters

Hard filters applied after retrieval (or as a pre-filter for buying mode):

```python
def apply_filters(self, product_indices: list[int], filters: dict) -> list[int]:
    """
    Removes products that violate hard constraints.
    Used strictly in BUYING mode.
    """
    kept = []
    for idx in product_indices:
        p = self.catalog.products[idx]
        if not self._passes_filters(p, filters):
            continue
        kept.append(idx)
    return kept

def _passes_filters(self, product: dict, filters: dict) -> bool:
    price = product.get("price") or 0.0

    if filters.get("price_max") and price > 0 and price > filters["price_max"]:
        return False
    if filters.get("price_min") and price > 0 and price < filters["price_min"]:
        return False

    if filters.get("brand"):
        product_store = product.get("store", "").lower()  # "store" not "brand"
        if filters["brand"].lower() not in product_store:
            return False

    # Reject products the user already said no to
    if product.get("parent_asin") in filters.get("rejected_asins", []):  # "parent_asin" not "asin"
        return False

    return True
```

### Hybrid Fusion (The Main Method)

**Reciprocal Rank Fusion (RRF)**: Combines two ranked lists without needing normalized scores.

Formula: `score(product) = 1/(rank_in_bm25 + k) + 1/(rank_in_vector + k)` where k=60 is a smoothing constant.

```python
def retrieve(
    self,
    query: str,
    filters: dict = None,
    mode: str = "hybrid",
    top_k: int = 50,
    buying_mode: bool = False,
) -> list[dict]:
    filters = filters or {}

    if mode == "bm25":
        raw = self.bm25_search(query, top_k=top_k * 4)
        indices = [i for i, _ in raw]
    elif mode == "vector":
        raw = self.vector_search(query, top_k=top_k * 4)
        indices = [i for i, _ in raw]
    else:  # hybrid (default)
        bm25_results  = self.bm25_search(query,  top_k=top_k * 4)
        vector_results = self.vector_search(query, top_k=top_k * 4)

        # Apply hard filters BEFORE fusion in buying mode
        # (diversity less important; correctness matters)
        if buying_mode and filters:
            bm25_idx   = self.apply_filters([i for i, _ in bm25_results],   filters)
            vector_idx = self.apply_filters([i for i, _ in vector_results], filters)
            # Rebuild with original scores
            bm25_results   = [(i, s) for i, s in bm25_results   if i in set(bm25_idx)]
            vector_results = [(i, s) for i, s in vector_results if i in set(vector_idx)]

        # RRF fusion
        rrf = {}
        K = 60
        for rank, (idx, _) in enumerate(bm25_results):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rank + K)
        for rank, (idx, _) in enumerate(vector_results):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (rank + K)

        sorted_items = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
        indices = [i for i, _ in sorted_items]

    # Apply soft filters in browsing mode (don't hard-filter, just deprioritize)
    if not buying_mode and filters:
        indices = self.apply_filters(indices, filters) + \
                  [i for i in indices if i not in set(self.apply_filters(indices, filters))]

    return [self.catalog.products[i] for i in indices[:top_k]]
```

### Building Query from State

The query fed to retrieval is built from accumulated slots + recent user message:

```python
def build_query(state: ConversationState) -> str:
    parts = []

    # Most important: category + use_case (semantic content)
    if state.slots.get("category"):   parts.append(state.slots["category"])
    if state.slots.get("use_case"):   parts.append(state.slots["use_case"])
    if state.slots.get("gender"):     parts.append(state.slots["gender"])
    if state.slots.get("style"):      parts.append(state.slots["style"])
    if state.slots.get("material"):   parts.append(state.slots["material"])
    if state.slots.get("color"):      parts.append(state.slots["color"])
    if state.slots.get("features"):   parts.extend(state.slots["features"])
    # Brand: useful for BM25 keyword match
    if state.slots.get("brand"):      parts.append(state.slots["brand"])

    # Also include the raw last user message for anything we missed
    if state.history:
        last_user = next(
            (m["content"] for m in reversed(state.history) if m["role"] == "user"),
            ""
        )
        parts.append(last_user)

    return " ".join(parts)


def build_filters(state: ConversationState) -> dict:
    """Extracts hard filter constraints from state."""
    filters = {}
    if state.slots.get("price_max"):  filters["price_max"] = state.slots["price_max"]
    if state.slots.get("price_min"):  filters["price_min"] = state.slots["price_min"]
    if state.slots.get("brand"):      filters["brand"]     = state.slots["brand"]
    if state.rejected_asins:          filters["rejected_asins"] = state.rejected_asins
    return filters
```

---

## Component 5: Orchestrator (`orchestrator.py`)

**Purpose**: Given the current state, decide the next action: `SEARCH`, `CLARIFY`, or `RECOMMEND`.

This is the brain of the system — it controls MTTC directly.

```python
class Orchestrator:
    # Tunable thresholds (tune these against the 200 sessions)
    MIN_SLOTS_BEFORE_SEARCH = 1     # at least one slot needed
    CANDIDATE_OVERLOAD_THRESHOLD = 500  # too many results → clarify
    TURNS_FORCE_SEARCH = 8          # at turn 8+, always search (avoid hitting limit)

    def decide(self, state: ConversationState, estimated_candidates: int) -> dict:
        """
        Returns a dict like:
        {"action": "SEARCH"}
        {"action": "CLARIFY", "missing_slots": ["use_case", "price_max"]}
        {"action": "RECOMMEND"}  # already have results, user asked to see them again
        """

        # Safety valve: near turn limit, always search
        if state.turn_count >= self.TURNS_FORCE_SEARCH:
            return {"action": "SEARCH", "reason": "near_limit"}

        # If the user just said "show me those" or "what do you have", just search
        if state.intent == "BROWSING":
            # Browsing: be more permissive, search early
            if state.slots.get("category") or state.turn_count >= 2:
                return {"action": "SEARCH", "diverse": True}
            else:
                return {"action": "CLARIFY", "missing_slots": ["category"]}

        # Buying mode: be more precise
        if state.intent == "BUYING":
            has_category = bool(state.slots.get("category"))

            if not has_category:
                return {"action": "CLARIFY", "missing_slots": ["category"]}

            # Check if query is still too vague
            if estimated_candidates > self.CANDIDATE_OVERLOAD_THRESHOLD:
                missing = self._get_priority_missing_slots(state)
                unasked = [s for s in missing if s not in state.asked_clarifications]
                if unasked:
                    return {"action": "CLARIFY", "missing_slots": unasked}

            # We have enough — search
            return {"action": "SEARCH"}

        # Unknown intent — need at least a category
        has_category = bool(state.slots.get("category"))
        if not has_category:
            return {"action": "CLARIFY", "missing_slots": ["category"]}

        if estimated_candidates > self.CANDIDATE_OVERLOAD_THRESHOLD:
            return {"action": "CLARIFY", "missing_slots": self._get_priority_missing_slots(state)}

        return {"action": "SEARCH"}

    def _get_priority_missing_slots(self, state: ConversationState) -> list[str]:
        # Ordered by how much they narrow down the search
        priority_order = ["use_case", "price_max", "gender", "brand", "color", "size"]
        return [s for s in priority_order if not state.slots.get(s)]
```

### Clarification Question Generation

```python
# Slot → question template
CLARIFICATION_QUESTIONS = {
    "category": "What type of item are you looking for? For example: shoes, clothing, or jewelry?",
    "use_case": "What will you be using {category} for — running, casual wear, formal events, or something else?",
    "price_max": "Do you have a budget in mind? What's the most you'd want to spend?",
    "gender":    "Is this for men, women, or are you open to unisex options?",
    "brand":     "Do you have a preferred brand, or should I show you the best options from any brand?",
    "color":     "Any color preference?",
    "size":      "What size do you need?",
}

def generate_clarification_question(missing_slots: list[str], state: ConversationState) -> str | None:
    """
    Picks the highest-priority unasked slot and returns a question.
    Returns None if we've already asked about everything.
    """
    for slot in missing_slots:
        if slot not in state.asked_clarifications:
            template = CLARIFICATION_QUESTIONS.get(slot, f"Could you tell me more about the {slot} you prefer?")
            question = template.format(**state.slots)
            state.asked_clarifications.add(slot)
            return question
    return None  # fallback: don't clarify, just search
```

---

## Component 6: Embedding Reranker (`retrieval.py` — `rerank()`)

**Purpose**: Take the top-50 candidates from retrieval and reorder them so the most relevant product is ranked first. This directly improves MRR.

**Approach: embedding cosine similarity (implemented in `retrieval.py`)**

Keyword/heuristic scoring was rejected because it hardcodes synonyms — "genuine leather" won't match `"leather"`, "navy" won't match `"blue"`, "Nike Air" won't match `"nike"`. Each miss requires a new rule. Embedding similarity handles all of this automatically.

The reranker:
1. Builds one preference query string from all accumulated slots + `state.last_query`
2. Encodes it once with the same `all-MiniLM-L6-v2` encoder already loaded in `Catalog`
3. Scores each candidate by `cosine_sim(preference_vec, product_vec)` — product vectors are looked up from the pre-computed `catalog_embeddings.npy` matrix (no extra inference per candidate)
4. Two small non-embedding adjustments only:
   - Gender: uses `details["Department"]` structural field — more reliable than embedding for this binary attribute
   - Rating: tiny tiebreaker (0–0.02 range, never overrides similarity signal)

```python
def rerank(self, candidates, state, top_k=10):
    if not candidates:
        return []

    rejected = set(state.rejected_asins or [])
    slots = state.slots

    # Build preference query from all accumulated slots
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
        return [p for p in candidates if p.get("parent_asin") not in rejected][:top_k]

    # Encode once
    q_vec = self.catalog.encoder.encode(
        pref_query, normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)

    scored = []
    for rank, p in enumerate(candidates):
        asin = p.get("parent_asin", "")
        if asin in rejected:
            continue
        idx = self.catalog.asin_to_idx.get(asin)
        sim = float(self.catalog.embeddings[idx] @ q_vec) if idx is not None else 0.0
        gender_adj = 0.0
        if slots.gender:
            prod_gender = Product.from_dict(p).gender_from_details()
            if prod_gender == slots.gender:
                gender_adj = 0.10
            elif prod_gender and prod_gender != slots.gender:
                gender_adj = -0.15
        rating_bonus = ((p.get("average_rating") or 0.0) / 5.0) * 0.02
        scored.append((sim + gender_adj + rating_bonus, rank, p))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [p for _, _, p in scored[:top_k]]
```

    return sorted(candidates, key=score, reverse=True)[:top_k]


def _category_match(product: dict, target_category: str) -> bool:
    target = target_category.lower()
    # Check in title
    if target in product.get("title", "").lower():
        return True
    # Check in flattened categories
    for cat_path in product.get("categories", []):
        if isinstance(cat_path, list):
            if any(target in c.lower() for c in cat_path):
                return True
    return False
```

---

## Component 7: Response Generator (`responder.py`)

**Purpose**: Format the ranked product list into a natural reply to the user.

```python
def generate_response(
    ranked_products: list[dict],
    state: ConversationState,
    action: str,
    clarification_question: str = None,
    llm=None
) -> str:

    if action == "CLARIFY":
        return clarification_question or "Could you tell me more about what you're looking for?"

    if not ranked_products:
        # Tell user we found nothing and suggest widening
        slots_used = list(state.slots.keys())
        return (
            f"I couldn't find products matching your criteria "
            f"({', '.join(slots_used[:3])}). "
            "Could you relax one of the constraints, like the brand or price range?"
        )

    top5 = ranked_products[:5]

    if llm:
        return _llm_response(top5, state, llm)
    else:
        return _template_response(top5, state)


def _template_response(products: list[dict], state: ConversationState) -> str:
    needs = _summarize_needs_brief(state)
    lines = [f"Here are the top {len(products)} results for {needs}:\n"]

    for i, p in enumerate(products, 1):
        title  = p.get("title", "Unknown product")[:70]
        price  = f"${p['price']:.2f}" if p.get("price") else "Price not listed"
        rating = f"⭐ {p['average_rating']:.1f} ({p.get('rating_number', 0)} reviews)" if p.get("average_rating") else ""
        brand  = p.get("store", "")      # "store" is the brand field
        asin   = p.get("parent_asin", "")

        lines.append(f"{i}. **{title}**")
        if brand:
            lines.append(f"   Brand: {brand}")
        lines.append(f"   {price}  {rating}")
        lines.append(f"   ID: {asin}")
        lines.append("")

    lines.append("Would you like more details, or should I refine these results?")
    return "\n".join(lines)


def _llm_response(products: list[dict], state: ConversationState, llm) -> str:
    needs = _summarize_needs_brief(state)
    product_info = "\n".join(
        f"- {p.get('title','')[:60]} | ${p.get('price','N/A')} | {p.get('brand','')}"
        for p in products
    )

    prompt = f"""You are a shopping assistant. The user wants: {needs}

Top matching products:
{product_info}

Write a friendly 2-3 sentence recommendation. Name the products and briefly say why
each fits the user's needs. Be concise — no bullet points, just natural prose."""

    return llm.complete(prompt, max_tokens=150)


def _summarize_needs_brief(state: ConversationState) -> str:
    s = state.slots
    parts = []
    if s.get("brand"):      parts.append(s["brand"])
    if s.get("category"):   parts.append(s["category"])
    if s.get("price_max"):  parts.append(f"under ${s['price_max']}")
    return " ".join(parts) or "your request"
```

---

## Component 8: Memory (`memory.py`)

**Purpose**: Track session context to enable multi-turn coherence, and optionally build user profiles across sessions.

```python
class SessionMemory:
    def __init__(self):
        self.messages = []          # [{"role": "user"|"assistant", "content": "..."}]
        self.search_history = []    # list of (query, slots_at_time, results_count)
        self.rejected_asins = []    # products user explicitly didn't want

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_recent_context(self, max_turns: int = 5) -> list[dict]:
        """Return the last N turns for LLM context."""
        return self.messages[-(max_turns * 2):]

    def record_rejection(self, asin: str):
        """User said they don't want this product."""
        if asin not in self.rejected_asins:
            self.rejected_asins.append(asin)

    def distill_to_summary(self, llm=None) -> str:
        """
        When conversation gets long (>6 turns), summarize earlier context
        to keep LLM prompts short.
        """
        if len(self.messages) <= 6 or not llm:
            return ""
        # Summarize turns 0..N-4, keep last 4 turns verbatim
        old_messages = self.messages[:-4]
        text = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)
        prompt = f"Summarize this shopping conversation in one sentence:\n{text}"
        return llm.complete(prompt, max_tokens=60)
```

**Long-term user profile** (optional, for extra points):
```python
import json, os

class UserProfile:
    """Persists preferences across sessions for a user_id."""

    def __init__(self, user_id: str, storage_dir: str = "./user_profiles"):
        self.user_id = user_id
        self.path = os.path.join(storage_dir, f"{user_id}.json")
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {"brand_counts": {}, "price_range": {}, "category_counts": {}}

    def update_from_session(self, state: ConversationState):
        s = state.slots
        if s.get("brand"):
            b = s["brand"]
            self.data["brand_counts"][b] = self.data["brand_counts"].get(b, 0) + 1
        if s.get("price_max"):
            self.data["price_range"]["last_max"] = s["price_max"]
        if s.get("category"):
            c = s["category"]
            self.data["category_counts"][c] = self.data["category_counts"].get(c, 0) + 1
        self._save()

    def get_priors(self) -> dict:
        """Returns soft priors to bias the initial state."""
        priors = {}
        if self.data["brand_counts"]:
            priors["preferred_brand"] = max(self.data["brand_counts"],
                                             key=self.data["brand_counts"].get)
        return priors

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f)
```

---

## Component 9: Main Agent (`agent.py`)

**Purpose**: The single class the competition evaluator instantiates. It wires all components together.

```python
class ShoppingAgent:
    """
    Public interface expected by the evaluator:
        agent = ShoppingAgent(catalog_path)
        response = agent.chat(user_message)   # called once per turn
        agent.reset()                         # called between sessions
    """

    def __init__(self, catalog_path: str, llm_client=None, config: dict = None):
        cfg = config or {}

        # Load catalog (expensive — done once at init, not per session)
        self.catalog = Catalog(catalog_path)

        # Components (stateless — safe to share across sessions)
        self.retrieval    = RetrievalPipeline(self.catalog)
        self.orchestrator = Orchestrator()
        self.reranker     = LLMReranker(llm_client)
        self.responder    = Responder(llm_client)
        self.llm          = llm_client

        # Session state (reset between sessions)
        self.state  = None
        self.memory = None
        self.reset()

    def reset(self):
        """Call this between sessions (the evaluator calls it)."""
        self.state  = ConversationState()
        self.memory = SessionMemory()

    def chat(self, user_message: str) -> str:
        """Main entry point — process one user turn and return the agent's reply."""

        # 1. Increment turn counter
        self.state.turn_count += 1

        # 2. Add user message to memory
        self.memory.add_message("user", user_message)
        self.state.history = self.memory.messages  # keep in sync

        # 3. Extract intent + slots from this message
        new_intent = classify_intent_fast(user_message)
        if new_intent is None:
            new_intent = classify_intent_llm(user_message, self.state.history, self.llm) \
                         if self.llm else "BUYING"  # safe default

        new_slots = extract_slots(user_message, self.state, self.llm)

        # 4. Update state (accumulate / override slots)
        self.state = update_state(self.state, new_slots, new_intent)

        # 5. Estimate how many products would match current query
        estimated_count = estimate_result_count(self.state, self.catalog)

        # 6. Decide action
        decision = self.orchestrator.decide(self.state, estimated_count)
        action = decision["action"]

        # 7. Execute action
        if action == "CLARIFY":
            question = generate_clarification_question(
                decision.get("missing_slots", []), self.state
            )
            if question:
                response = question
            else:
                # We've clarified enough, fall through to search
                action = "SEARCH"

        if action == "SEARCH":
            query   = build_query(self.state)
            filters = build_filters(self.state)

            buying_mode = (self.state.intent == "BUYING")
            candidates = self.retrieval.retrieve(
                query=query,
                filters=filters,
                mode="hybrid",
                top_k=50,
                buying_mode=buying_mode,
            )
            self.state.last_query = query

            # Rerank
            ranked = self.reranker.rerank(candidates, self.state, top_k=10)
            self.state.last_candidates = ranked

            # Generate response
            response = self.responder.generate(ranked, self.state, action="RECOMMEND")

        # 8. Store assistant reply in memory
        self.memory.add_message("assistant", response)

        return response
```

---

## Component 10: Config (`config.py`)

```python
# Retrieval
BM25_CANDIDATE_POOL   = 200      # how many BM25 results to fetch
VECTOR_CANDIDATE_POOL = 200      # how many vector results to fetch
HYBRID_TOP_K          = 50       # candidates passed to reranker
RERANK_TOP_K          = 10       # final results returned to user
RRF_K                 = 60       # reciprocal rank fusion smoothing constant

# Orchestration
MIN_SLOTS_BEFORE_SEARCH      = 1    # must have at least 1 slot
CANDIDATE_OVERLOAD_THRESHOLD = 500  # over this → ask clarification
TURNS_FORCE_SEARCH           = 8    # at this turn, always search

# Model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # ~80MB, fast, good quality
EMBEDDINGS_CACHE = "catalog_embeddings.npy"

# Evaluation
HIT_AT_K = 10   # evaluate Hit Rate at top 10
```

---

## Phase 1 Integration Sequence

Build in this order. Test each piece before moving on.

```
Day 1 Morning — Everyone together
├── Clone and run the starter kit
├── Run the evaluator on baseline
├── Record: Hit@10 = ?, MRR = ?, MTTC = ?
├── Read 20 sessions from the 200 to understand patterns
└── Split work

Day 1 Afternoon
├── Person 1: catalog.py + retrieval.py
│   Goal: retrieve() returns 50 candidates that include the target product
│   Test: manually check 10 sessions, is target in top 50?
│
├── Person 2: intent.py + state.py
│   Goal: update_state() correctly accumulates slots across 3-turn test dialogs
│   Test: write 5 synthetic conversations, print state after each turn
│
└── Person 3: orchestrator.py + responder.py
    Goal: decide() makes sensible SEARCH/CLARIFY choices
    Test: feed various states, verify decisions make sense

Day 1 Evening — Integration
└── Wire everything together in agent.py
    Run evaluator
    Record new: Hit@10 = ?, MRR = ?, MTTC = ?

Day 2 — Optimization
└── Look at where the system fails, fix those specific issues
```

---

## Phase 2: Multi-Agent (Only If Metrics Stall)

Look at your Day 1 numbers first:

| If you see this... | Do this... |
|--------------------|-----------|
| Hit@10 < 60% | Fix retrieval — better query building, tune BM25/vector weights |
| Hit@10 > 80% but MRR < 0.4 | Fix reranking — better LLM prompt or scoring |
| MRR > 0.5 but MTTC > 5 | Fix orchestration — ask fewer/smarter clarification questions |
| All metrics good | Add multi-agent structure for judge's "architecture" score |

If you go multi-agent, the refactor is minimal — each existing module becomes an "agent" with a clean interface:

```
Controller (agent.py)
    ↓ calls
ConversationAgent  → wraps intent.py + state.py
    ↓ returns updated state

SearchAgent        → wraps retrieval.py
    ↓ returns candidates

RankingAgent       → wraps reranker.py + responder.py
    ↓ returns ranked response
```

The Controller decides which agents to call and in what order — the logic from `orchestrator.py` moves here.

---

## LLM Options (No API Key Required Path)

The competition says a paid LLM is NOT required. Here are your options:

| Option | Cost | Speed | Quality | Notes |
|--------|------|-------|---------|-------|
| **Heuristic only** | Free | Fast | OK | Use regex + scoring functions from this plan |
| **Ollama + llama3.2:3b** | Free | Medium | Good | Run locally, ~2GB RAM |
| **Ollama + llama3.1:8b** | Free | Slower | Better | ~5GB RAM |
| **Claude API** | Paid | Fast | Best | Use `claude-haiku-4-5-20251001` for speed/cost |
| **OpenAI GPT-4o-mini** | Paid | Fast | Good | Cheap, reliable |

**Recommended minimum viable path**: Use heuristic scoring for reranking + template responses. The BM25/vector retrieval quality matters far more than the response quality for the metrics being evaluated.

---

## What Each Metric Rewards

### Hit Rate@10 (Coverage)
- Is the target product in your top 10?
- Improved by: better retrieval (BM25 tuning, vector quality, query building)
- If a product isn't in top 50, the reranker can't save you

### MRR (Precision)
- How high is the target product ranked?
- Improved by: better reranking (LLM quality, scoring formula)
- Even with Hit@10 = 100%, MRR can be low if target is always at rank 10

### MTTC (Efficiency)
- How many turns before user finds the product?
- Improved by: asking fewer, smarter clarification questions
- **Key insight**: a system that searches on turn 2 with moderate confidence beats one that clarifies 4 times then searches on turn 5

### Optimization Priority Order
1. Get Hit@10 to 70%+ (retrieval must be solid)
2. Get MTTC below 4 (orchestration must be efficient)
3. Improve MRR above 0.5 (reranking matters here)

---

## Common Failure Modes to Watch For

1. **Category mismatch**: User says "sneakers", catalog has "Athletic Shoes" — build synonym map
2. **Price filter too strict**: No products under $30 → return empty → ask user to raise budget
3. **Over-clarification**: Asking 4 questions before searching → MTTC suffers → lower the threshold
4. **BM25 keyword miss**: "jogging shoes" won't match products titled "Running Footwear" → use vector search more
5. **Hit limit**: Agent reaches 10 turns without finding product → score = 0 → always search at turn 8
