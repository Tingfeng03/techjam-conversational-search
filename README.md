# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

---

## Project Overview

Our solution is a multi-stage retrieval and dialogue agent that finds a user's target product from a 50,000-item clothing, shoes, and jewellery catalogue through natural conversation — entirely without LLM API calls at runtime.

**Why no LLM API at runtime?** LLM queries add 1–4 seconds of latency per turn and hit rate limits fast (free-tier APIs cap at ~15 RPM, making a 200-session evaluation take hours). Instead, we use pre-trained transformer models that run locally on CPU after a one-time Hugging Face download — no API keys, no rate limits, deterministic results.

The pipeline per turn:
1. **Intent detection** — classify the message kind (buy, browse, override, answer, no-preference, rejection) before updating state
2. **Constraint parsing** — deterministic rules map evaluator constraint strings to typed slots (`"Department: Womens"` → `gender=women`, `"budget around $45"` → `price_max=45`)
3. **Hybrid retrieval** — BM25 + bi-encoder vector search (200 candidates each), merged via Reciprocal Rank Fusion (k=60), hard-filtered to top 100
4. **Three-pass reranking** — adaptive structural+semantic blend (100→40), then cross-encoder with natural-language query (40→10)
5. **Adaptive clarification** — retrieve first, then ask the question with highest `coverage × diversity × novelty` over the actual visible candidates
6. **Intent override recovery** — when the user overrides mid-session, skip the clarification turn and search immediately

**Final TechnicalScore: 0.7744** (baseline weak starter: 0.107 → our improvements: 0.7744)

| Scenario | HitRate@10 | MTTC |
|---|---|---|
| Buying | 0.925 | 3.0 turns |
| Browsing | 0.9375 | 3.64 turns |
| Intent Override | 0.900 | 5.3 turns |
| Boundary | 0.800 | 5.5 turns |
| **Overall** | **0.920** | **3.875 turns** |

---

## Setup and Installation

Python 3.10 or later required.

**Quick install (global):**
```bash
pip install -r requirements.txt
```

**Isolated install (recommended — won't affect your other projects):**
```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Note: `sentence-transformers` pulls in PyTorch automatically — first install is ~1GB.

---

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

---

## Steps to Reproduce Our Results

**1. Install dependencies**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Download and place the catalog**
```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

**3. Build the embedding cache** (runs automatically on first evaluation; ~3 minutes on CPU)

The first run of the evaluator encodes all 50,000 products and writes `catalog_embeddings.npy`. Subsequent runs load the cache in ~5 seconds. Delete the `.npy` file only if the catalog changes.

**4. Run the evaluator**
```bash
PYTHONPATH=. python evaluator/local_evaluator.py \
    --dataset data/public_set.jsonl \
    --catalog data/catalog.jsonl
```

Results are written to `results.json`. Expected output on the public set:

```
hit_rate_at_10: 0.920
mrr:            0.573
mttc:           3.875
TechnicalScore: 0.7744
```

**Key files changed from the starter:**

| File | What changed |
|---|---|
| `starter/agent.py` | Wires all components; OVERRIDE intercept; post-retrieval clarification |
| `starter/orchestration/orchestration.py` | `Orchestrator` with `rank_clarifications()`; adaptive decide() |
| `starter/orchestration/responder.py` | Updated `Responder` interface |
| `state.py` | Full `constraint_to_slots()` parser; `_extract_slots_from_category()` |
| `retrieval.py` | Three-pass reranker; structural scoring; natural-language CE query; CE doc enrichment |
| `catalog.py` | Added cross-encoder model load |

---

## Limitations and What We Would Improve Given More Time

**Current limitations:**

- **Constraint parser is brittle** — `constraint_to_slots()` in `state.py` is a deterministic regex rule set. It works well for the evaluator's structured constraint strings but would struggle with free-text user input ("something warm for the slopes" → use_case=skiing is not extracted).

- **`estimate_result_count` ignores active filters** — the category-index estimate used by `Orchestrator.decide()` doesn't account for brand, price, or color filters. This means the `CANDIDATE_OVERLOAD_THRESHOLD` signal is noisy and the orchestrator cannot reliably detect when the candidate pool is already narrow enough to stop asking questions.

- **Boundary and late-session no-preference handling** — in sessions where the user keeps responding "no preference", the system keeps asking clarifying questions until `TURN_FORCE_SEARCH=10`, wasting turns. A low-yield detection mechanism (many questions asked, few slots filled) would switch to pure search earlier — but tracking which slots were filled from initial disclosure vs. question answers requires additional state.

- **Cross-encoder runs on CPU** — scoring 40 pairs per turn takes ~200ms. On GPU this would be ~10ms, making it practical to widen the cross-encoder pool further.

**Given more time, we would:**

1. **Complete the schema-driven migration** (`GENERALITY_STATUS.md` details this fully): replace the fixed `Slots` dataclass with a generic `ConstraintSet`, route constraint extraction through the schema-declared extractor, and move structural reranking behind schema-provided matchers. This would allow adding a new product domain (e.g. electronics with `battery_life`, `screen_size`) by writing only an `AttributeSpec` and a parser — no changes to orchestration, retrieval, or reranking code.

2. **Use actual post-retrieval candidate count** to decide when to stop clarifying — count the retrieved candidates after filters instead of estimating from the category index.

3. **Track per-turn slot gain** — record which slots were empty before asking a question and filled after the answer. Use this signal to detect low-yield users and switch to pure search earlier, reducing MTTC for boundary sessions.

4. **Tune the cross-encoder document** further — the CE doc currently includes `Department | Color | Material`. Adding `Size` and `Use Case` fields for sessions where those are the discriminating constraints could improve late-turn MRR.

---

## Team Member Contributions

| Member | Contributions |
|---|---|
| **Person 1** | `catalog.py` — product loading, BM25 index, bi-encoder embedding cache, cross-encoder model load; `retrieval.py` — hybrid BM25+vector retrieval, RRF fusion, category pre-filtering, three-pass reranker, structural scoring, natural-language CE query, CE doc enrichment |
| **Person 2** | `domain_schema.py` — schema-driven architecture (`AttributeSpec`, `DomainSchema`); schema integration into `Orchestrator`; `GENERALITY_STATUS.md` |
| **Person 3** | `state.py` — full constraint parsing pipeline (`constraint_to_slots`, `_extract_slots_from_category`), intent detection integration; `starter/orchestration/orchestration.py` — `Orchestrator.decide()`, `rank_clarifications()`, adaptive clarification scoring; `starter/agent.py` — full agent wiring, OVERRIDE intercept; `starter/orchestration/responder.py` |

---

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
