from __future__ import annotations

from interfaces import ConversationState, Filters, OrchestratorDecision

CLARIFICATION_QUESTIONS = {
  "category": "What type of item are you looking for? For example: shoes, clothing, or jewelry?",
  "use_case": "What will you be using {category} for — running, casual wear, formal events, or something else?",
  "price_max": "Do you have a budget in mind? What's the most you'd want to spend?",
  "gender":    "Is this for men, women, or are you open to unisex options?",
  "brand":     "Do you have a preferred brand, or should I show you the best options from any brand?",
  "color":     "Any color preference?",
  "size":      "What size do you need?",
}

def generate_clarification(missing_slots: list[str], state: ConversationState) -> str | None:
  """
  Picks the highest-priority unasked slot and returns a question.
  Returns None if we've already asked about everything.
  """
  for slot in missing_slots:
    if slot not in state.asked_clarifications:
      template = CLARIFICATION_QUESTIONS.get(slot, f"Could you tell me more about the {slot} you prefer?")
      # Slots is a dataclass, not a mapping.  Use all attributes so templates
      # remain safe even when the referenced value has not been filled yet.
      values = vars(state.slots)
      question = template.format_map(_MissingSlotValues(values))
      state.asked_clarifications.add(slot)
      return question
  return None 


class _MissingSlotValues(dict):
  def __missing__(self, key: str) -> str:
    return ""


def build_query(state: ConversationState) -> str:
  """Build the text query used to retrieve products for the current state.

  Prices and rejected products are intentionally omitted: they are exact
  constraints handled by :func:`build_filters`, not useful search terms.
  """
  slots = state.slots
  parts: list[str] = []

  # These fields describe what should be found.  Keep the terms in their
  # natural form so either BM25 or semantic retrieval can consume the query.
  for value in (
    slots.category,
    slots.brand,
    slots.gender,
    slots.use_case,
    slots.style,
    slots.material,
    slots.color,
    slots.size,
  ):
    if isinstance(value, str) and value.strip():
      parts.append(value.strip())

  for feature in slots.features:
    if isinstance(feature, str) and feature.strip():
      parts.append(feature.strip())

  if isinstance(state.last_query, str) and state.last_query.strip():
    parts.append(state.last_query.strip())

  # Avoid accidentally overweighting a term mentioned in multiple slots.
  unique_parts = list(dict.fromkeys(parts))
  return " ".join(unique_parts) if unique_parts else "clothing shoes jewelry"


def build_hyde_query(state: ConversationState) -> str:
  """Build a product-title-style phrase for vector search.

  Produces natural phrasing (e.g. "Nike women's blue cotton shoes for running")
  rather than a keyword bag, which better matches how catalog product titles
  are written and how sentence-transformers embed them.
  Falls back to build_query() when no slots are filled.
  """
  slots = state.slots
  phrase: list[str] = []

  if slots.brand:
    phrase.append(slots.brand)

  if slots.gender:
    g = slots.gender.lower()
    if g in ("men", "man", "male"):
      phrase.append("men's")
    elif g in ("women", "woman", "female"):
      phrase.append("women's")
    else:
      phrase.append(slots.gender)

  for val in (slots.color, slots.material, slots.style):
    if isinstance(val, str) and val.strip():
      phrase.append(val.strip())

  if slots.category:
    phrase.append(slots.category)

  if slots.use_case:
    phrase.append(f"for {slots.use_case}")

  if slots.size:
    phrase.append(f"size {slots.size}")

  for feat in (slots.features or [])[:2]:
    if isinstance(feat, str) and feat.strip():
      phrase.append(feat.strip())

  if isinstance(state.last_query, str) and state.last_query.strip():
    phrase.append(state.last_query.strip())

  result = " ".join(dict.fromkeys(p for p in phrase if p))
  return result.strip() or build_query(state)


def build_filters(state: ConversationState) -> Filters:
  """Extract hard, structured constraints for retrieval.

  Budget and previously rejected products are always hard constraints.  Brand
  is hard only for a buying request; in browsing mode it remains a preference
  for retrieval/reranking rather than excluding all other products.
  """
  slots = state.slots

  def valid_price(value: object) -> float | None:
    if value is None:
      return None
    try:
      price = float(value)
    except (TypeError, ValueError):
      return None
    return price if price >= 0 else None

  brand = slots.brand.strip() if isinstance(slots.brand, str) else None
  if not brand:
    brand = None

  # Preserve order while removing duplicate rejections.  Retrieval can safely
  # use this list as an exclusion set without mutating conversation state.
  rejected_asins = list(dict.fromkeys(
    asin for asin in state.rejected_asins if isinstance(asin, str) and asin
  ))

  return Filters(
    price_min=valid_price(slots.price_min),
    price_max=valid_price(slots.price_max),
    brand=brand if state.intent == "BUYING" else None,
    rejected_asins=rejected_asins,
  )


class Orchestrator:
  TURN_FORCE_SEARCH = 6  # always search when turn >= this threshold
  CANDIDATE_OVERLOAD_THRESHOLD = 150  # too many results, so clarify

  def decide(
    self,
    state: ConversationState,
    estimated_candidates: int,
  ) -> OrchestratorDecision:

    if state.turn_count >= self.TURN_FORCE_SEARCH:
      return OrchestratorDecision(
        action="SEARCH",
        reason="near_limit",
        diverse=(state.intent == "BROWSING")
      )

    has_category = bool(state.slots.category)

    if not has_category:
      return OrchestratorDecision(
          action="CLARIFY",
          missing_slots=["category"],
          reason="no_searchable_category"
      )

    unasked = self._get_priority_missing_slots(state)

    # In buying mode with a hard constraint already disclosed, search sooner.
    threshold = self.CANDIDATE_OVERLOAD_THRESHOLD
    if state.intent == "BUYING" and (state.slots.brand or state.slots.price_max):
      threshold = 500

    # searchable request, but too broad — search and ask one more question
    if (estimated_candidates > threshold and unasked):
      return OrchestratorDecision(
          action="SEARCH_AND_CLARIFY",
          missing_slots=unasked,
          reason="searchable_but_broad",
          diverse=(state.intent == "BROWSING")
      )

    return OrchestratorDecision(
      action="SEARCH",
      reason="sufficiently_specific",
      diverse=(state.intent == "BROWSING")
    )

  def _get_priority_missing_slots(
    self, 
    state: ConversationState
  ) -> list[str]:
    if state.intent == "BROWSING":
      priority_order = [
        "use_case",
        "style",
        "features",
        "gender",
        "color",
      ]
    else:
      priority_order = [
        "use_case",
        "price_max",
        "gender",
        "brand",
        "color",
        "material",
        "size",
      ]

    return [
      slot for slot in priority_order
      if not getattr(state.slots, slot) and slot not in state.asked_clarifications
    ]
