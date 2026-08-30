from __future__ import annotations

import math
import re
from collections import Counter
from typing import Mapping

from interfaces import ConversationState, Filters, OrchestratorDecision


_ATTRIBUTE_ORDER = ("features", "material", "color", "style", "size", "use_case", "price_max", "brand")
_ATTRIBUTE_TO_PUBLIC = {"features": "feature", "price_max": "budget"}
_ATTRIBUTE_SLOTS = {
  "features": ("features",), "material": ("material",), "color": ("color",),
  "style": ("style", "gender"), "size": ("size",), "use_case": ("use_case",),
  "price_max": ("price_max", "price_min"), "brand": ("brand",),
}
_MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
_COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)
_USE_CASE_RE = re.compile(r"\b(hiking|running|gym|winter|outdoor|work)\b", re.I)
_PROFILE_WORDS = {
  "features": {"comfort", "fit", "durability", "quality", "warmth", "weather", "waterproof"},
  "material": {"material", "fabric", "leather", "cotton", "wool"},
  "color": {"color", "colour"}, "style": {"style", "fit", "formal", "casual", "athletic"},
  "size": {"size", "fit", "width"}, "use_case": {"use", "weather", "warmth", "outdoor", "running", "hiking"},
  "price_max": {"price", "budget", "value", "affordable"}, "brand": {"brand"},
}

CLARIFICATION_QUESTIONS = {
  "category": "What type of item are you looking for? For example: shoes, clothing, or jewelry?",
  "use_case": "What will you be using {category} for — running, casual wear, formal events, or something else?",
  "price_max": "Do you have a budget in mind? What's the most you'd want to spend?",
  "gender":    "Is this for men, women, or are you open to unisex options?",
  "brand":     "Do you have a preferred brand, or should I show you the best options from any brand?",
  "color":     "Any color preference?",
  "size":      "What size do you need?",
  "features":  "Which feature matters most to you, such as comfort, durability, warmth, or something else?",
  "other":     "Is there another specific requirement I should prioritize?",
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


def _norm(value: object) -> str:
  return " ".join(str(value or "").lower().split())


def _price_bucket(value: object) -> str | None:
  try:
    price = float(value)
  except (TypeError, ValueError):
    return None
  if price <= 0:
    return None
  if price < 25:
    return "<25"
  if price < 50:
    return "25-50"
  if price < 100:
    return "50-100"
  return "100+"


def _product_values(product: dict, attribute: str) -> set[str]:
  """Extract observable metadata for clarification scoring only."""
  details = product.get("details") or {}
  text = " ".join([
    str(product.get("title") or ""),
    *(str(value) for value in (product.get("features") or [])),
    *(str(value) for value in (product.get("categories") or [])),
  ])
  if attribute == "features":
    return {_norm(value) for value in (product.get("features") or []) if _norm(value)}
  if attribute == "material":
    values = {_norm(details.get("Material"))} if details.get("Material") else set()
    values.update(match.group(1).lower() for match in _MATERIAL_RE.finditer(text))
    return {value for value in values if value}
  if attribute == "color":
    values = {_norm(details.get("Color"))} if details.get("Color") else set()
    values.update(match.group(1).lower() for match in _COLOR_RE.finditer(text))
    return {value for value in values if value}
  if attribute == "style":
    values = {
      _norm(value) for key, value in details.items()
      if any(word in str(key).lower() for word in ("department", "style", "fit", "sleeve", "neck"))
      and _norm(value)
    }
    values.update(_norm(value) for value in (product.get("categories") or []) if _norm(value))
    return values
  if attribute == "size":
    value = details.get("Size") or details.get("Sizing")
    return {_norm(value)} if _norm(value) else set()
  if attribute == "use_case":
    return {match.group(1).lower() for match in _USE_CASE_RE.finditer(text)}
  if attribute == "price_max":
    bucket = _price_bucket(product.get("price"))
    return {bucket} if bucket else set()
  if attribute == "brand":
    value = product.get("store") or details.get("Brand")
    return {_norm(value)} if _norm(value) else set()
  return set()


def _diversity(value_sets: list[set[str]]) -> float:
  non_empty = [values for values in value_sets if values]
  if len(non_empty) < 2:
    return 0.0
  if any(len(values) > 1 for values in non_empty):
    distances: list[float] = []
    for index, left in enumerate(non_empty):
      for right in non_empty[index + 1:]:
        union = left | right
        distances.append(1.0 - len(left & right) / len(union) if union else 0.0)
    return sum(distances) / len(distances) if distances else 0.0
  counts = Counter(next(iter(values)) for values in non_empty)
  if len(counts) <= 1:
    return 0.0
  entropy = -sum((count / len(non_empty)) * math.log(count / len(non_empty)) for count in counts.values())
  return entropy / math.log(len(counts))


def _profile_multiplier(state: ConversationState, attribute: str) -> float:
  tags = {_norm(tag) for tag in ((state.user_profile or {}).get("preference_tags") or []) if _norm(tag)}
  words = _PROFILE_WORDS.get(attribute, set())
  aligned = any(tag in words or any(word in tag for word in words) for tag in tags)
  return 1.1 if aligned else 1.0


class Orchestrator:
  """Choose when to search, then rank questions from visible candidates."""

  TURN_FORCE_SEARCH = 10
  CANDIDATE_OVERLOAD_THRESHOLD = 500
  MIN_ATTRIBUTE_COVERAGE = 0.20

  def __init__(self, answer_priors: Mapping[str, float] | None = None) -> None:
    self.answer_priors: dict[str, float] = {}
    for attribute, value in (answer_priors or {}).items():
      try:
        number = float(value)
      except (TypeError, ValueError):
        continue
      if math.isfinite(number) and number >= 0:
        self.answer_priors[str(attribute)] = number

  def decide(
    self,
    state: ConversationState,
    estimated_candidates: int,
  ) -> OrchestratorDecision:

    if not state.slots.category:
      return OrchestratorDecision(
        action="CLARIFY", missing_slots=["category"], reason="no_searchable_category",
      )
    if state.turn_count >= self.TURN_FORCE_SEARCH:
      return OrchestratorDecision(
        action="SEARCH", reason="turn_limit", diverse=(state.intent == "BROWSING"),
      )
    breadth = "broad" if estimated_candidates > self.CANDIDATE_OVERLOAD_THRESHOLD else "specific"
    return OrchestratorDecision(
      action="SEARCH_AND_CLARIFY",
      reason=f"search_then_adapt:{breadth}",
      diverse=(state.intent == "BROWSING"),
    )

  def rank_clarifications(self, state: ConversationState, candidates: list[dict]) -> list[str]:
    """Return internal slot names in descending question-value order."""
    if not candidates:
      return []
    scored: list[tuple[float, int, str]] = []
    for order, attribute in enumerate(_ATTRIBUTE_ORDER):
      if self._asked_or_declined(state, attribute):
        continue
      value_sets = [_product_values(product, attribute) for product in candidates]
      coverage = sum(bool(values) for values in value_sets) / len(value_sets)
      if coverage < self.MIN_ATTRIBUTE_COVERAGE:
        continue
      diversity = _diversity(value_sets)
      novelty = 0.6 if self._known(state, attribute) else 1.0
      public_name = _ATTRIBUTE_TO_PUBLIC.get(attribute, attribute)
      prior = self.answer_priors.get(public_name, 1.0)
      value = prior * coverage * (1.0 + diversity) * novelty * _profile_multiplier(state, attribute)
      scored.append((value, -order, attribute))
    scored.sort(reverse=True)
    if scored:
      return [attribute for _, _, attribute in scored]
    if state.turn_count >= 6 and "other" not in state.asked_clarifications:
      return ["other"]
    return []

  def clarification_diagnostic(self, state: ConversationState, candidates: list[dict], attribute: str) -> str:
    value_sets = [_product_values(product, attribute) for product in candidates]
    coverage = sum(bool(values) for values in value_sets) / len(value_sets) if value_sets else 0.0
    diversity = _diversity(value_sets)
    novelty = 0.6 if self._known(state, attribute) else 1.0
    public_name = _ATTRIBUTE_TO_PUBLIC.get(attribute, attribute)
    prior = self.answer_priors.get(public_name, 1.0)
    value = prior * coverage * (1.0 + diversity) * novelty * _profile_multiplier(state, attribute)
    return f"adaptive:{public_name}:value={value:.3f}:coverage={coverage:.2f}:diversity={diversity:.2f}"

  @staticmethod
  def _known(state: ConversationState, attribute: str) -> bool:
    return any(bool(getattr(state.slots, slot, None)) for slot in _ATTRIBUTE_SLOTS[attribute])

  @staticmethod
  def _asked_or_declined(state: ConversationState, attribute: str) -> bool:
    if any(slot in state.asked_clarifications for slot in _ATTRIBUTE_SLOTS[attribute]):
      return True
    public_name = _ATTRIBUTE_TO_PUBLIC.get(attribute, attribute)
    return public_name in state.asked_clarifications

  @staticmethod
  def _get_priority_missing_slots(state: ConversationState) -> list[str]:
    """Compatibility helper for callers of the former static policy."""
    return [slot for slot in _ATTRIBUTE_ORDER if not getattr(state.slots, slot, None)]

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
        "size",
        "material",
        "color",
      ]

    return [
      slot for slot in priority_order
      if not getattr(state.slots, slot) and slot not in state.asked_clarifications
    ]
