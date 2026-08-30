from __future__ import annotations

from typing import Optional


_SLOT_TO_ATTRIBUTE = {
  "price_min": "budget",
  "price_max": "budget",
  "features": "feature",
}
_VALID_ATTRIBUTES = {
  "category",
  "material",
  "color",
  "size",
  "style",
  "brand",
  "budget",
  "feature",
  "use_case",
  "other",
}


def _ask_attribute(asked_slot: Optional[str]) -> Optional[str]:
  """Map internal slot names to the evaluator's allowed attribute names."""
  if not asked_slot:
    return None
  attribute = _SLOT_TO_ATTRIBUTE.get(asked_slot, asked_slot)
  return attribute if attribute in _VALID_ATTRIBUTES else "other"


def _recommendations(ranked_products: list[dict]) -> list[dict]:
  """Return the first ten unique, non-empty parent ASINs in ranking order."""
  recommendations: list[dict] = []
  seen: set[str] = set()

  for product in ranked_products:
    if not isinstance(product, dict):
      continue
    parent_asin = product.get("parent_asin")
    if not isinstance(parent_asin, str) or not parent_asin or parent_asin in seen:
      continue
    seen.add(parent_asin)
    recommendations.append({"parent_asin": parent_asin})
    if len(recommendations) == 10:
      break

  return recommendations


def _search_message(ranked_products: list[dict]) -> str:
  if not ranked_products:
    return "I couldn't find products matching your criteria. Try relaxing a constraint."

  first_product = ranked_products[0]
  if not isinstance(first_product, dict):
    return "Here are the closest matches I found."

  title = first_product.get("title")
  return (
    f"Here are the closest matches I found. Top result: {title}."
    if isinstance(title, str) and title.strip()
    else "Here are the closest matches I found."
  )


class Responder:
  def __init__(self, llm_client=None) -> None:
    self.llm_client = llm_client

  def format_response(
    self,
    ranked_products: list[dict],
    action: str,
    clarification_question: Optional[str] = None,
    asked_slot: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
  ) -> dict:
    """Format one evaluator-compatible response without mutating session state.

    ``CLARIFY`` asks one structured question and returns no products.
    ``SEARCH`` returns ranked recommendations. ``SEARCH_AND_CLARIFY`` does both
    in the same turn, so a miss produces a useful simulator reply next turn.
    """
    # The evaluator only accepts non-negative integer usage values.
    usage = {
      "prompt_tokens": max(0, int(prompt_tokens)),
      "completion_tokens": max(0, int(completion_tokens)),
    }

    if action == "CLARIFY":
      return {
        "message": clarification_question or "Could you tell me more about what you're looking for?",
        "ask_attribute": _ask_attribute(asked_slot),
        "recommendations": [],
        "usage": usage,
      }

    if action in {"SEARCH_CLARIFY", "SEARCH_AND_CLARIFY"}:
      question = clarification_question or "Could you tell me more about what you're looking for?"
      return {
        "message": f"{_search_message(ranked_products)} {question}",
        "ask_attribute": _ask_attribute(asked_slot),
        "recommendations": _recommendations(ranked_products),
        "usage": usage,
      }

    if action == "FALLBACK":
      return {
        "message": "I couldn't find products matching your criteria. Try relaxing a constraint.",
        "ask_attribute": None,
        "recommendations": [],
        "usage": usage,
      }

    # Treat SEARCH and an unknown action as a search response rather than
    # returning malformed output to the evaluator.
    return {
      "message": _search_message(ranked_products),
      "ask_attribute": None,
      "recommendations": _recommendations(ranked_products),
      "usage": usage,
    }
