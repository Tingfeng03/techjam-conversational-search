"""
interfaces.py — Do NOT edit without group discussion.

This file is the contract between all three team members.
Person 3 (Orchestrator) imports from here and wires agent.py against stubs.
Person 1 and 2 implement their own files to match these exact signatures.

IMPORTANT — Evaluator API (from docs/agent_api_contract.json):
    The evaluator calls:
        agent.reset(session_id: str, user_profile: dict)
        agent.respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict

    respond() MUST return exactly:
        {
            "message":         str,           # text reply shown to user
            "ask_attribute":   str | None,    # slot name if clarifying, else None
                               # REQUIRED field — use null, not omit
                               # allowed values: "category", "material", "color",
                               # "size", "style", "brand", "budget", "feature",
                               # "use_case", "other", null
            "recommendations": [              # list of products, best first (max 100, scored on top 10)
                {"parent_asin": str},         # "score" field is IGNORED by the evaluator — only parent_asin matters
                ...
            ],
            "usage": {                        # optional but include it — token counts (set 0 if no LLM)
                "prompt_tokens": int,
                "completion_tokens": int,
            }
        }

    The catalog field is "parent_asin", NOT "asin". Double-check your catalog loading.
    top_k is always 10 in the competition. recommendations maxItems is 100 — return up to 50 safely.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# SHARED DATA SCHEMAS
# =============================================================================

@dataclass
class Product:
    """
    One product from the 50k catalog.
    Person 1 returns lists of these from retrieve() and rerank().
    Person 3 passes them to responder.

    VERIFIED against data/catalog.jsonl — actual field names and types:
        parent_asin    — unique ID e.g. "B07K34RX5J"  (NOT "asin" — that field does not exist)
        title          — product name string  (2 products have empty title — guard with `or ""`)
        description    — LIST of strings (NOT a single string!) e.g. ["Long description..."]
                         EMPTY LIST in 47.8% of products (23,887 / 50,000) — don't rely on it
        features       — list of strings e.g. ["Spandex", "Made in USA", "Lightweight..."]
                         EMPTY LIST in 10.4% of products (5,219 / 50,000)
        price          — float, string, OR null. Three cases:
                           null string   : 39,473 products (78.9%) — no price listed
                           string        : 117 products — garbage like "—" or "from 12.99"
                           float         : remaining ~10,410 products, range $0.01–$4119
                           zero (float 0): 1 product — treat as missing
                         USE safe_price() helper below, NOT raw p["price"]
        store          — brand/store name e.g. "Spirit Hoops"  (NOT "brand" — that field does not exist)
                         null in 314 products — from_dict returns "" in that case
        categories     — FLAT list of strings e.g. ["Clothing, Shoes & Jewelry", "Women", "Jewelry"]
                         (NOT a list of lists — the whole path is already one flat list)
                         First element is always "Clothing, Shoes & Jewelry" (49,990/50,000)
                         NEVER empty — all 50k products have at least one category
        details        — dict of extra metadata, NEVER null (1,670 products have empty dict {})
                         Useful keys (verified counts):
                           "Department"  — 43,582 products — gender e.g. "Womens", "Mens", "Girls"
                           "Brand"       — 2,328 products  — secondary brand when store is null
                           "Material"    — 2,069 products
                           "Color"       — 2,439 products
                           "Size"        — 925 products
        average_rating — float 1.0–5.0  (NEVER null — all 50k products have this)
        rating_number  — int > 0  (NEVER null or zero — all 50k products have this)

    WRONG fields used in initial plan (do NOT use these — they will KeyError):
        "asin"        → use "parent_asin"
        "brand"       → use "store"  (or details["Brand"] as fallback)
        "rating"      → use "average_rating"
        "rating_count"→ use "rating_number"
        description as str → it is a list, call .description_text() to get a string
    """
    parent_asin: str
    title: str
    price: Optional[float]              # use safe_price() — raw value may be str or 0
    store: str                          # brand equivalent — "" when null (314 products)
    categories: list[str]               # flat list of strings, NOT list of lists
    features: list[str]                 # may be empty list (10.4% of products)
    description: list[str]             # may be empty list (47.8% of products)
    details: dict                       # never null; useful keys: Department, Brand, Material, Color
    average_rating: float               # never null, range 1.0–5.0
    rating_number: int                  # never null or zero

    @staticmethod
    def from_dict(d: dict) -> "Product":
        """Convert a raw catalog dict to a Product."""
        return Product(
            parent_asin=d.get("parent_asin", ""),
            title=d.get("title") or "",
            price=Product._safe_price(d.get("price")),
            store=d.get("store") or "",             # 314 products have store=null
            categories=d.get("categories") or [],
            features=d.get("features") or [],
            description=d.get("description") or [], # list of strings
            details=d.get("details") or {},
            average_rating=d.get("average_rating") or 0.0,
            rating_number=d.get("rating_number") or 0,
        )

    @staticmethod
    def _safe_price(raw) -> Optional[float]:
        """
        Convert raw price to float or None.
        Handles: null → None, float → float, str "—" → None, str "from 12.99" → None,
                 float 0 → None (treat zero as missing).
        """
        if raw is None:
            return None
        try:
            val = float(raw)
            return val if val > 0 else None
        except (ValueError, TypeError):
            return None

    def description_text(self) -> str:
        """Join description list into one searchable string."""
        if isinstance(self.description, list):
            return " ".join(self.description)
        return str(self.description)

    def gender_from_details(self) -> Optional[str]:
        """
        Extract gender from details["Department"] — present in 43,582 / 50,000 products.
        Returns "women", "men", "girls", "boys", "unisex", or None.
        """
        dept = (self.details.get("Department") or "").lower()
        if not dept:
            return None
        if "women" in dept or "female" in dept or "ladies" in dept:
            return "women"
        if "men" in dept or "male" in dept:
            return "men"
        if "girls" in dept:
            return "girls"
        if "boys" in dept:
            return "boys"
        if "unisex" in dept or "adult" in dept:
            return "unisex"
        return None

    def brand(self) -> str:
        """
        Best available brand string — tries store first, falls back to details["Brand"].
        Returns "" if neither is available.
        """
        return self.store or self.details.get("Brand", "")


@dataclass
class Slots:
    """
    All structured shopping constraints extracted from the conversation so far.
    Person 2 fills this in. Person 1 and 3 read from it.

    All fields default to None = "user hasn't specified this yet".
    """
    category:  Optional[str]        = None  # "running shoes", "women's dress"
    brand:     Optional[str]        = None  # "Nike", "Levi's"
    price_min: Optional[float]      = None  # 0.0
    price_max: Optional[float]      = None  # 100.0
    color:     Optional[str]        = None  # "blue", "black"
    size:      Optional[str]        = None  # "M", "10", "8.5"
    gender:    Optional[str]        = None  # "men", "women", "unisex"
    use_case:  Optional[str]        = None  # "running", "hiking", "formal"
    features:  list[str]            = field(default_factory=list)  # ["waterproof", "lightweight"]
    style:     Optional[str]        = None  # "athletic", "formal", "casual"
    material:  Optional[str]        = None  # "leather", "cotton"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != []}


@dataclass
class ConversationState:
    """
    Full state of one session. Person 3 reads this to make decisions.
    Person 2 updates and returns it each turn.
    Person 3 passes it to retrieve(), rerank(), and respond().

    Lifecycle:
        - Created fresh at session start (agent.reset())
        - Updated by update_state() each turn
        - Read by decide(), retrieve(), rerank(), respond()
        - Never shared across sessions
    """
    # --- session metadata ---
    session_id:   str = ""
    turn_count:   int = 0       # the evaluator passes turn number directly (1-indexed)
    max_turns:    int = 10      # hard competition limit — do not exceed

    # --- user profile (passed by evaluator at reset time) ---
    # VERIFIED: actual fields in public_set.jsonl (all 200 sessions):
    #   purchase_frequency   — always "3-4 prior purchases" in public set
    #   average_prior_rating — float e.g. 5.0  (never null in public set, despite contract allowing null)
    #   rating_style         — one of: "usually positive", "mixed", "critical"
    #   preference_tags      — list of strings e.g. ["fit", "comfort", "durability"]
    #   summary              — one sentence e.g. "Prior purchases emphasize fit..."
    user_profile: dict = field(default_factory=dict)

    # --- what the user wants ---
    # VERIFIED scenario counts in public set (200 sessions):
    #   "buying"         →  80 sessions — specific product, hard constraints
    #   "browsing"       →  80 sessions — exploring, soft preferences
    #   "intent_override"→  30 sessions — switches request mid-session (hardest)
    #   "boundary"       →  10 sessions — edge case, user says "no preference"
    # difficulty_bucket: easy=80, medium=90, hard=30 (in session JSON but NOT passed to agent)
    #
    # CRITICAL — intent_override hit timing (verified from evaluator source):
    #   override fires at turn 3 or 4 (randomly assigned per session)
    #   recommendations on turns BEFORE the override turn can NEVER score a hit
    #   even if the correct product is returned — the evaluator blocks it
    #   → in intent_override sessions, earliest possible hit = turn 3
    #   → don't waste clarification turns early; search fast and pivot on the override message
    intent: str = "UNKNOWN"     # "BUYING" | "BROWSING" | "UNKNOWN"
    slots:  Slots = field(default_factory=Slots)

    # --- conversation history ---
    # Each entry: {"role": "user" | "assistant", "content": "..."}
    history: list[dict] = field(default_factory=list)

    # --- search state ---
    last_candidates: list[dict] = field(default_factory=list)  # raw dicts from catalog
    last_query: str = ""

    # --- clarification tracking ---
    asked_clarifications: set = field(default_factory=set)  # slot names already asked about

    # --- rejection tracking ---
    rejected_asins: list[str] = field(default_factory=list)  # ASINs user said no to


@dataclass
class Filters:
    """
    Hard constraints used by retrieve() to exclude products.
    Built from ConversationState.slots by Person 3 before calling retrieve().

    Only set fields that are HARD requirements (violating = wrong product).
    Do NOT set fields for soft preferences (those are handled by reranker).

    When to use each field:
        price_max      — always set if slots.price_max is not None
        price_min      — set if slots.price_min is not None
        brand          — set ONLY in BUYING mode (in BROWSING, brand is a soft preference)
                         NOTE: Person 1 must match this against product["store"], not product["brand"]
                         e.g. filters.brand = "Nike" → check if "nike" in product["store"].lower()
        rejected_asins — always pass state.rejected_asins here
    """
    price_max:      Optional[float]     = None
    price_min:      Optional[float]     = None
    brand:          Optional[str]       = None
    rejected_asins: list[str]           = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None and v != []}


@dataclass
class OrchestratorDecision:
    """
    What the orchestrator decided to do next.
    Returned by decide(). Consumed by agent.py to drive control flow.

    action values:
        "SEARCH"   — retrieve + rerank + respond with products
        "CLARIFY"  — ask user a question, no retrieval this turn
        "FALLBACK" — could not find products, tell user to relax constraints
    """
    action: str                         # "SEARCH" | "CLARIFY" | "FALLBACK" | "SEARCH_AND_CLARIFY"
    missing_slots: list[str] = field(default_factory=list)  # only set when action="CLARIFY"
    reason: str = ""                    # short debug string e.g. "over_general", "near_limit"
    diverse: bool = False               # True = browsing mode, prioritise variety in retrieval


# =============================================================================
# PERSON 1 — STUBS  (retrieval.py + reranker.py)
# =============================================================================

def retrieve(
    query: str,
    filters: Filters,
    top_k: int = 50,
    buying_mode: bool = False,
    category: str | None = None,
) -> list[dict]:
    """
    Search the 50k catalog and return the top-K most relevant raw product dicts.

    Args:
        query       : Free-text search string built from accumulated slots
                      e.g. "Nike running shoes lightweight men"
                      Person 3 builds this from ConversationState using build_query()

        filters     : Hard constraints. Products violating these MUST be excluded.
                      e.g. Filters(price_max=100.0, brand="Nike")

        top_k       : How many candidates to return. Default 50.
                      The reranker will cut this down to 10 later.

        buying_mode : If True,  apply filters STRICTLY (exclude all violations).
                      If False, apply filters SOFTLY (penalise violations, don't exclude).
                      Person 3 sets this to (state.intent == "BUYING").

        category    : Optional category string (e.g. slots.category) to restrict
                      search to matching products. Narrows from 50k to ~1k products,
                      improving recall within the right category. Defaults to None
                      (search full catalog). Pass state.slots.category when available.

    Returns:
        List of up to top_k raw product dicts from the catalog.
        Each dict has at minimum: parent_asin, title, price, store, categories, features.
        NOTE: brand is "store" in the catalog — use product["store"], NOT product["brand"].
        Sorted by relevance descending (best match first).
        Returns [] if nothing found.

    Example:
        candidates = retrieve(
            query="Nike running shoes lightweight men",
            filters=Filters(price_max=100.0, brand="Nike"),
            top_k=50,
            buying_mode=True,
            category="Running Shoes",
        )
        # candidates[0] = {"parent_asin": "B07XYZ", "title": "Nike Air Zoom...", "price": 89.99, ...}
    """
    # STUB — Person 1 replaces this body
    return []


def rerank(
    candidates: list[dict],
    state: ConversationState,
    top_k: int = 10,
) -> list[dict]:
    """
    Re-order the candidate pool so the most relevant product is ranked first.
    This directly improves MRR.

    Args:
        candidates  : Raw product dicts from retrieve(). Up to 50 items.

        state       : Full conversation state. Use state.slots to score relevance.
                      Also use state.rejected_asins to penalise rejected products.

        top_k       : How many results to return. Default 10 (evaluator uses Hit@10).

    Returns:
        List of up to top_k raw product dicts, best match first.
        Must be a SUBSET of candidates (no new products injected).
        Returns [] if candidates is empty.

    Example:
        ranked = rerank(candidates, state, top_k=10)
        # ranked[0] should be the product the user is most likely to buy
    """
    # STUB — Person 1 replaces this body
    return candidates[:top_k]


# =============================================================================
# PERSON 2 — STUBS  (intent.py + state.py)
# =============================================================================

def update_state(
    state: ConversationState,
    message: str,
) -> ConversationState:
    """
    Process one user message and return an updated ConversationState.
    This is the main function Person 2 implements.

    It must handle three scenarios internally:
        1. Information Accumulation — user adds constraints
           "Also make it blue" → state.slots.color = "blue"  (keep everything else)

        2. Intent Override — user switches request entirely
           "Actually forget shoes, show me jackets"
           → clear old category/use_case slots, set new ones
           → keep price_max if user didn't change it

        3. Rejection — user says no to shown products
           "I don't want that first one"
           → add ASIN to state.rejected_asins

    Args:
        state   : Current conversation state BEFORE this turn.
                  Do NOT mutate — return a new/updated copy.

        message : Raw user message string for this turn.
                  e.g. "I need Nike running shoes under $100"

    Returns:
        Updated ConversationState with:
            - state.intent updated if a clearer signal was found
            - state.slots updated with any new or overridden constraints
            - state.history has the user message appended
              (assistant reply is appended by agent.py after response is generated)

    Example input → output:
        # Turn 1
        state = ConversationState()
        state = update_state(state, "I need running shoes")
        # state.intent = "BUYING"
        # state.slots.category = "running shoes"

        # Turn 2
        state = update_state(state, "Under $100, Nike please")
        # state.intent = "BUYING"          (unchanged)
        # state.slots.category = "running shoes"  (accumulated)
        # state.slots.price_max = 100.0    (new)
        # state.slots.brand = "Nike"       (new)

        # Turn 3 — override
        state = update_state(state, "Actually make it Adidas")
        # state.slots.brand = "Adidas"     (overridden)
        # state.slots.category = "running shoes"  (kept)
        # state.slots.price_max = 100.0    (kept)
    """
    # STUB — Person 2 replaces this body
    state.history.append({"role": "user", "content": message})
    return state


def estimate_result_count(
    state: ConversationState,
    catalog,                    # Catalog object from catalog.py
) -> int:
    """
    Estimate how many products would match the current slots WITHOUT doing a real search.
    Used by the orchestrator to decide if the query is too vague.

    Args:
        state   : Current state. Uses state.slots.category mainly.
        catalog : The Catalog object (has catalog.category_index dict).

    Returns:
        Estimated integer count of matching products.
        Return len(catalog.products) (~50000) if no category is set yet.
        Return 1 minimum (never 0 — avoid division by zero in caller).

    Example:
        # state.slots.category = "running shoes"
        count = estimate_result_count(state, catalog)
        # count = 342  (there are 342 running shoe products)
        # orchestrator sees 342 < 500 threshold → decides to SEARCH
    """
    # STUB — Person 2 replaces this body
    if not state.slots.category:
        return 50000
    return 100   # dummy


# =============================================================================
# PERSON 3 — STUBS  (orchestrator.py + responder.py)
# These are what Person 3 actually implements — shown here for Person 1+2 clarity
# =============================================================================

def decide(
    state: ConversationState,
    estimated_candidates: int,
) -> OrchestratorDecision:
    """
    Decide the next action given the current state.

    Args:
        state               : Full conversation state after update_state().
        estimated_candidates: Result from estimate_result_count().

    Returns:
        OrchestratorDecision with action = "SEARCH" | "CLARIFY" | "FALLBACK"

    Example:
        decision = decide(state, estimated_candidates=4200)
        # decision.action = "CLARIFY"
        # decision.missing_slots = ["use_case", "price_max"]
        # decision.reason = "over_general"
    """
    # STUB — Person 3 replaces this body
    return OrchestratorDecision(action="SEARCH")


def generate_clarification(
    missing_slots: list[str],
    state: ConversationState,
) -> Optional[str]:
    """
    Return a question to ask the user for the highest-priority unasked slot.

    Args:
        missing_slots : Slot names we need, in priority order.
                        e.g. ["use_case", "price_max", "gender"]
        state         : Used to check state.asked_clarifications (avoid repeats)
                        and to fill in slot values in templates
                        e.g. "What will you be using {category} for?"

    Returns:
        A question string to send to the user.
        Returns None if all missing_slots have already been asked about
        (caller should fall through to SEARCH in that case).

    Example:
        q = generate_clarification(["use_case", "price_max"], state)
        # "What will you be using running shoes for — running, casual wear, or something else?"
    """
    # STUB — Person 3 replaces this body
    return "Could you tell me more about what you're looking for?"


def build_query(state: ConversationState) -> str:
    """
    Build a free-text search query string from accumulated slots.
    Called by Person 3 in agent.py before calling retrieve().

    Args:
        state : Uses state.slots and state.history[-1] (last user message).

    Returns:
        A single string to pass to retrieve() as the query argument.

    Example:
        # state.slots = {category: "running shoes", brand: "Nike",
        #                gender: "men", price_max: 100}
        query = build_query(state)
        # "running shoes Nike men lightweight"
        # (price_max is a filter, not a query term)
    """
    # STUB — Person 3 replaces this body
    parts = []
    s = state.slots
    if s.category:  parts.append(s.category)
    if s.brand:     parts.append(s.brand)
    if s.gender:    parts.append(s.gender)
    if s.use_case:  parts.append(s.use_case)
    if s.color:     parts.append(s.color)
    parts.extend(s.features)
    return " ".join(parts) if parts else "clothing shoes jewelry"


def build_filters(state: ConversationState) -> Filters:
    """
    Extract hard constraints from state into a Filters object.
    Called by Person 3 before calling retrieve().

    Args:
        state : Uses state.slots and state.rejected_asins.

    Returns:
        Filters object with only the hard constraints set.

    Example:
        # state.slots.price_max = 100.0, state.slots.brand = "Nike"
        # state.intent = "BUYING"
        filters = build_filters(state)
        # Filters(price_max=100.0, brand="Nike", rejected_asins=[])
    """
    # STUB — Person 3 replaces this body
    s = state.slots
    return Filters(
        price_max=s.price_max,
        price_min=s.price_min,
        brand=s.brand if state.intent == "BUYING" else None,
        rejected_asins=state.rejected_asins,
    )


# removed because it is inaccurate

# =============================================================================
# QUICK REFERENCE — Turn flow in agent.py
# The evaluator calls: agent.reset(session_id, user_profile)
#                      agent.respond(session_id, user_message, turn, top_k) -> dict
# =============================================================================
#
#   def respond(self, session_id, user_message, turn, top_k) -> dict:
#
#       1.  self.state.turn_count = turn   # evaluator passes turn number directly
#
#       2.  self.state = update_state(self.state, user_message)         # Person 2
#
#       3.  est = estimate_result_count(self.state, self.catalog)       # Person 2
#
#       4.  decision = decide(self.state, est)                          # Person 3
#
#       5a. if decision.action == "CLARIFY":
#               question = generate_clarification(                      # Person 3
#                   decision.missing_slots, self.state)
#               if question:
#                   asked = decision.missing_slots[0]  # highest-priority unasked slot
#                   return respond([], self.state, "CLARIFY", question,  # Person 3  → dict
#                                  asked_slot=asked)
#               else:
#                   decision.action = "SEARCH"    # fallthrough
#
#       5b. if decision.action == "SEARCH":
#               query      = build_query(self.state)                    # Person 3
#               filters    = build_filters(self.state)                  # Person 3
#               candidates = retrieve(                                  # Person 1
#                   query, filters, top_k=50,
#                   buying_mode=(self.state.intent == "BUYING"))
#               ranked     = rerank(candidates, self.state, top_k=10)   # Person 1
#               return respond(ranked, self.state, "SEARCH")            # Person 3  → dict
#
#   NOTE: respond() returns a dict, not a string. agent.respond() returns that dict directly.
