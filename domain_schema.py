"""Typed, domain-owned metadata used by the conversational orchestrator.

The core orchestration policy depends only on this module's contracts.  A
domain may provide richer parser/matcher implementations without changing the
conversation loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


ProductValues = Callable[[dict], set[str]]


class ConstraintExtractor(Protocol):
    def extract(self, text: str, current: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ClothingDeterministicExtractor:
    """Adapter for the existing deterministic parser.

    The lazy import avoids a state/schema import cycle while making the
    extractor replaceable by a future structured LLM implementation.
    """
    def extract(self, text: str, current: Mapping[str, Any]) -> Mapping[str, Any]:
        from state import constraint_to_slots
        scalars, features = constraint_to_slots(text)
        result = dict(scalars)
        if features:
            result["features"] = features
        return result


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    value_kind: str = "scalar"
    aliases: tuple[str, ...] = ()
    public_attribute: str | None = None
    clarification_template: str | None = None
    clarification_priority: int = 0
    required_for_search: bool = False
    queryable: bool = True
    hard_in_intents: frozenset[str] = frozenset()
    profile_terms: frozenset[str] = frozenset()
    product_values: ProductValues | None = None

    @property
    def public_name(self) -> str:
        return self.public_attribute or self.name


@dataclass(frozen=True)
class DomainSchema:
    domain_id: str
    default_query: str
    attributes: tuple[AttributeSpec, ...]
    catch_all_attribute: str | None = None
    extractor: ConstraintExtractor | None = None
    product_id_field: str = "parent_asin"
    _by_name: dict[str, AttributeSpec] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        names: set[str] = set()
        aliases: set[str] = set()
        by_name: dict[str, AttributeSpec] = {}
        for spec in self.attributes:
            if not spec.name or spec.name in names:
                raise ValueError(f"duplicate/empty attribute: {spec.name!r}")
            names.add(spec.name)
            by_name[spec.name] = spec
            for alias in spec.aliases:
                key = alias.strip().lower()
                if key in aliases or key in names:
                    raise ValueError(f"duplicate attribute alias: {alias!r}")
                aliases.add(key)
        if self.catch_all_attribute and self.catch_all_attribute not in names:
            raise ValueError("catch_all_attribute must name a registered attribute")
        object.__setattr__(self, "_by_name", by_name)

    def get(self, name: str) -> AttributeSpec | None:
        return self._by_name.get(name)

    def resolve(self, name: str) -> AttributeSpec | None:
        key = (name or "").strip().lower()
        for spec in self.attributes:
            if key == spec.name.lower() or key in {a.lower() for a in spec.aliases}:
                return spec
        return None

    def clarifiable(self) -> tuple[AttributeSpec, ...]:
        return tuple(s for s in self.attributes if s.clarification_template)

    def validate(self) -> None:
        required = [s for s in self.attributes if s.required_for_search]
        if len(required) > 1:
            raise ValueError("a domain may have at most one required search attribute")
        if not self.default_query.strip():
            raise ValueError("default_query cannot be empty")


def clothing_schema() -> DomainSchema:
    """Schema metadata for the existing clothing catalog.

    Product extraction remains supplied by the legacy adapter for now; this
    registry is the stable seam used by orchestration and future domains.
    """
    schema = DomainSchema(
        domain_id="clothing_shoes_jewelry",
        default_query="clothing shoes jewelry",
        attributes=(
            AttributeSpec("category", aliases=("product type",), public_attribute="category", required_for_search=True),
            AttributeSpec("features", value_kind="multi", aliases=("feature",), public_attribute="feature", clarification_template="Which feature matters most to you?"),
            AttributeSpec("material", aliases=("fabric",), public_attribute="material", clarification_template="Do you have a material preference?"),
            AttributeSpec("color", aliases=("colour",), public_attribute="color", clarification_template="Any color preference?"),
            AttributeSpec("style", aliases=("fit",), public_attribute="style", clarification_template="What style or fit do you prefer?"),
            AttributeSpec("size", aliases=("width", "sizing"), public_attribute="size", clarification_template="What size do you need?"),
            AttributeSpec("use_case", aliases=("use", "occasion"), public_attribute="use_case", clarification_template="What will you be using this for?"),
            AttributeSpec("price_max", value_kind="number_range", aliases=("budget",), public_attribute="budget", clarification_template="Do you have a budget in mind?"),
            AttributeSpec("brand", public_attribute="brand", clarification_template="Do you have a preferred brand?"),
        ),
        catch_all_attribute="features",
        extractor=ClothingDeterministicExtractor(),
    )
    schema.validate()
    return schema


DEFAULT_SCHEMA = clothing_schema()
