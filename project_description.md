# Project Description

## How our solution addresses the problem statement

The challenge is a conversational shopping agent: given a 50,000-item clothing, shoes, and jewellery catalogue, find the user's target product within at most 10 turns of dialogue. The evaluator scores each session on three axes — whether the target appeared in our top-10 recommendations (HitRate@10), how highly it was ranked (MRR), and how quickly the system found it (Efficiency based on turns taken).

Our solution is a multi-stage retrieval and dialogue agent that operates entirely without LLM API calls at runtime.

### Why no LLM API calls?

An LLM query adds 1–4 seconds of latency per turn and costs tokens on every session. In a real shopping assistant with hundreds of concurrent users, this compounds into both a UX problem (users wait) and a cost problem (every clarification question and every rerank pass hits the API). More critically for this competition, the evaluator runs 200 sessions back-to-back — at 15 requests-per-minute on a free-tier API, that would take hours and still hit rate limits mid-evaluation.

Instead, we use pre-trained transformer models that are downloaded once from Hugging Face and run entirely locally on CPU. The bi-encoder encodes the query in one forward pass; the cross-encoder scores 40 candidate pairs in a single batch. No network calls, no API keys, no rate limits — the full 200-session evaluation completes in under 5 minutes.

This also means the system is deterministic and auditable: given the same user message, it always produces the same result, which makes debugging and improvement iteration fast.

---

### How a turn works

**1. Intent detection**

Every user message is classified (initial buy request, browsing, intent override, direct answer, no-preference, rejection) before any state is updated. This lets the system react differently to "Actually, ignore that — I need leather" versus "Yes, I prefer cotton".

**2. Constraint parsing**

A deterministic rule-based extractor maps the evaluator's structured constraint strings directly into typed slots:
- `"Department: Womens"` → `gender=women`
- `"budget around $45"` → `price_max=45`
- `"polyester"` → `material=polyester`

The first version of this was a stub that left every slot empty all session; fixing it was the single largest score jump (+0.193 TechnicalScore).

**3. Hybrid retrieval**

BM25 and a bi-encoder vector search run in parallel, each returning 200 candidates. Reciprocal Rank Fusion (k=60) merges the two rank lists into one, normalising the incompatible score scales. Hard filters (price, brand, rejected products) are applied after fusion, yielding a top-100 candidate pool. BM25 handles exact brand/model queries; vector search handles semantic similarity; neither alone is sufficient.

**4. Three-pass reranking**

- *Pass 1* blends a structural constraint-match score with bi-encoder similarity using adaptive weights that shift from semantic-heavy (early turns, few slots filled) to constraint-heavy (later turns, many slots filled). This produces a top-40 shortlist.
- *Pass 2* runs a cross-encoder over all 40 pairs. The cross-encoder query is a natural-language product-title phrase (`"Nike women's blue shoes for running"`) rather than a keyword bag, to match the model's MS-MARCO training distribution. The cross-encoder document includes the product's `Department`, `Color`, and `Material` detail fields so the model can match hard constraints directly. Final score = CE score + structural score as a tiebreaker.

**5. Adaptive clarification**

The system always retrieves first, then selects which clarifying question to ask by scoring each unasked attribute against the actual retrieved candidates. The attribute with the highest `coverage × diversity × novelty` score is asked — coverage means "most candidates have a value for this", diversity means "those values are varied", novelty means "this would actually tell us something new". Asking the highest-information question over the visible candidate set consistently outperforms a fixed priority order.

**6. Intent override recovery**

When the evaluator sends an override message mid-session, the system detects it before updating state and skips the clarification question for that turn, returning results immediately. This recovers a full turn of efficiency in override sessions (hit rate 0.300 → 0.900 for that scenario).

---

## What can be improved

The current codebase introduces a schema-driven architecture (`domain_schema.py`) that partially decouples the clarification policy from clothing-specific field names. The `Orchestrator` already accepts a `DomainSchema` object and uses it to select clarification attributes, compute candidate coverage and value diversity, and generate question templates — meaning a new product domain can add attributes without touching the clarification algorithm.

However, several layers still contain clothing-shaped assumptions that a future migration would address:

**Generic constraint storage**
`ConversationState.slots` is backed by a fixed `Slots` dataclass with hardcoded fields (`brand`, `color`, `material`, etc.). A new domain attribute like `battery_life` cannot be stored without extending that dataclass. Replacing it with a generic `ConstraintSet` dict-like structure would let any schema-declared attribute be stored and retrieved without code changes.

**Schema-routed constraint extraction**
The constraint parser in `state.py` is a clothing-specific deterministic rule set. The `DomainSchema` interface already defines an extractor contract, but the state updater does not yet select the extractor from the caller-supplied schema. Once it does, swapping in a different extractor (or an LLM-backed one for ambiguous domains) becomes a one-line change.

**Schema-driven query and filter construction**
`build_query()` and `build_filters()` read clothing fields directly. A schema-provided list of query-contributing attributes and filter clauses would make these functions domain-agnostic, so a new domain's attributes automatically flow into retrieval without additional changes.

**Schema-driven reranking**
Structural scoring in the retrieval pipeline checks brand, color, material, gender, price, use-case, size, and features by name. Moving these checks behind schema-provided product-value extractors and matchers would allow the same reranking pass to work for any domain.

Once those four migration phases are complete, adding a genuinely new product domain — say, consumer electronics — would require only: an `AttributeSpec` per attribute, a constraint parser/normalizer, a product-value extractor, and optional clarification metadata. The orchestration, retrieval, and reranking code would need no changes.

---

## Development tools used

- **VS Code** with the Excalidraw extension (architecture diagrams)
- **Claude Code** (AI-assisted development and analysis)
- Python 3.9

---

## APIs used

None at runtime — the system is fully offline. No OpenAI, Google, or Anthropic API calls during evaluation. All transformer models run locally via `sentence-transformers` after a one-time Hugging Face download.

---

## Libraries and frameworks used

| Library | Version | Purpose |
|---|---|---|
| `sentence-transformers` | ≥3.0 | Bi-encoder (`multi-qa-MiniLM-L6-cos-v1`) for vector search and pass-1 reranking; cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for final reranking pass |
| `bm25s` | ≥0.2 | Inverted-index BM25 retrieval (~500× faster than `rank_bm25` via sparse index) |
| `numpy` | ≥1.26 | Pre-computed L2-normalised product embeddings cached to `.npy`; dot-product scoring at query time |
| `tqdm` | ≥4.66 | Progress display during evaluation |

---

## Datasets and assets used

- **Product catalogue** — 50,000 clothing, shoes, and jewellery products from the TechJam competition dataset (`data/catalog.jsonl`), each with title, features, details (Department, Color, Material, Brand), categories, price, store, and ratings.
- **Evaluation sessions** — 200 labelled public sessions (`data/public_set.jsonl`) covering four scenario types: buying (specific target), browsing (exploratory), intent override (mid-session preference change), and boundary (no-preference edge cases).
- **Pre-trained models** (downloaded from Hugging Face at first run, then cached locally):
  - `multi-qa-MiniLM-L6-cos-v1` — bi-encoder for dense retrieval and pass-1 reranking
  - `cross-encoder/ms-marco-MiniLM-L-6-v2` — cross-encoder for final reranking pass
