# Schema-Driven Generality: Current Status

## Purpose

The recent refactor introduces a typed domain-schema seam, but it is a
partial migration rather than a fully schema-independent orchestration
pipeline. This document records exactly what is generalized today and what
still requires migration.

## What is generalized now

`domain_schema.py` defines reusable `AttributeSpec` and `DomainSchema`
contracts. A schema can provide:

- Attribute names, aliases, value kinds, and public response mappings.
- Clarification templates and priorities.
- Required-for-search and queryable metadata.
- Candidate-product value extractors.
- A deterministic constraint-extractor interface for future extractor types.

The orchestrator accepts a schema and uses it for:

- Selecting clarification attributes.
- Calculating candidate coverage and value diversity.
- Resolving public attribute names.
- Selecting schema-provided product-value callbacks.
- Generating schema-provided clarification text.

For example, a synthetic domain can define an attribute without adding a new
branch to the clarification algorithm:

```python
electronics_schema = DomainSchema(
    domain_id="electronics",
    default_query="electronics",
    attributes=(
        AttributeSpec("topic", required_for_search=True),
        AttributeSpec(
            name="battery_life",
            clarification_template="How important is battery life?",
            product_values=lambda p: {
                p["battery_life"]
            } if p.get("battery_life") else set(),
        ),
    ),
)

orchestrator = Orchestrator(schema=electronics_schema)
```

The existing clothing behavior remains the default through the clothing
schema registered as `DEFAULT_SCHEMA`. Existing state and response behavior
remain compatible with the current evaluator contract.

## What is not generalized yet

Adding a new schema alone does **not** yet onboard a new catalog end to end.
The following components still contain clothing-shaped assumptions:

### State storage

`ConversationState.slots` is still backed by the fixed `Slots` dataclass in
`interfaces.py`. A new attribute such as `battery_life` cannot be stored
without extending that dataclass or replacing it with a generic constraint
store.

### Constraint extraction

The active extractor is still the clothing deterministic parser in `state.py`.
The schema exposes an extractor interface, but the state updater does not yet
select an arbitrary extractor from a caller-supplied domain schema.

### Query construction

`build_query()` still reads clothing fields directly (`category`, `brand`,
`gender`, `use_case`, `style`, `material`, `color`, `size`, and `features`).
New schema attributes are not automatically included in retrieval queries.

### Filtering

`build_filters()` and `Filters` still represent price and brand explicitly.
Generic hard/soft filter clauses have not yet replaced these fields.

### Retrieval and reranking

The retrieval pipeline still contains clothing-specific product-field access
and structural scoring for brand, color, material, gender, price, use case,
size, and features. Schema-provided matchers are not yet used for all ranking
or filtering decisions.

### Category scoping and result counts

Category lookup and result-count estimation still use the catalog's existing
category index directly rather than a domain-provided category adapter.

## Current extension workflow

For a clothing-only change, the safest current workflow is:

1. Add or modify an `AttributeSpec` in the clothing schema.
2. Add the corresponding deterministic parsing rule in `state.py`.
3. Extend `Slots` if the value needs a new stored field.
4. Extend `build_query()`, `build_filters()`, or retrieval scoring if the
   attribute affects search behavior.
5. Add positive, negative, ambiguity, and clarification-ranking tests.

For a genuinely new domain, a schema entry must currently be accompanied by
the equivalent state and retrieval changes listed above.

## Target extension workflow after the remaining migration

The intended end state is:

```text
DomainSchema
  ├── deterministic or LLM extractor
  ├── generic constraint storage
  ├── query-term adapters
  ├── filter clauses
  ├── product-value extractors
  ├── matchers
  └── category resolver

Generic state → Generic orchestrator → Generic retrieval/reranking
```

At that point, adding `battery_life` would require only:

1. An `AttributeSpec`.
2. A parser/normalizer.
3. A product-value extractor and matcher.
4. Optional clarification and public-response metadata.

The generic state, orchestration, query, filtering, and ranking code would not
need to be edited.

## Remaining migration phases

1. Replace fixed `Slots` storage with a generic `ConstraintSet`, retaining a
   compatibility view for existing callers.
2. Route state updates through the schema-selected extractor.
3. Replace fixed query construction with schema-declared query terms.
4. Replace `Filters` price/brand fields with generic filter clauses.
5. Move structural reranking and candidate metadata extraction behind schema
   callbacks.
6. Move category scoping and result-count estimation behind a schema/catalog
   adapter.
7. Add synthetic non-clothing end-to-end tests and run evaluator regression
   checks.

## Interpretation

The current implementation is a safe architectural first tranche: it removes
schema coupling from clarification policy and creates the contracts needed for
future extraction implementations. It should not yet be described as a
fully plug-and-play multi-domain system. That claim becomes accurate only
after state, query construction, filtering, and retrieval/reranking complete
the migration above.
