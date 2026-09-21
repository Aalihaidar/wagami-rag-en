"""A read-only profile of the knowledge base, built once at startup from the corpus itself.

It serves two jobs, and both need to stay true to the data without anyone maintaining a copy:

* the query-understanding prompt describes the knowledge base to the model -- how many rows,
  which item types, how each is organised, and every value the limited-value fields take
  (app/agent/prompts.py renders this);
* browsing ("what's on the menu?", "show me the drinks") is answered directly from it, with no
  search and no second model call (app/agent/browse.py).

Vocabulary. A row's `category_path` runs from a top-level **group** (its first item) down to the
row's **category** (its last item). For every menu item the last item equals the `category`
field. Guests call a group a "category" and a category a "sub-category"; this module uses the
data's own words so `category` always means the field of that name.
"""

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

# Fields read from each row to build the catalog (a subset of the collection's properties).
CATALOG_PROPERTIES = [
    "item_type",
    "name",
    "category",
    "category_slug",
    "category_path",
    "dietary_tags",
    "allergens_contains",
    "allergens_may_contain",
    "is_gluten_free_listed",
    "portion_unit",
    "servings",
    # what an item card and an item listing show
    "slug",
    "description",
    "ingredients",
    "price_gbp",
    "image",
]

# The fields whose values form a small closed set, so the prompt can list every one of them.
# (Free text, ids, prices and the nutrition figures are described, not enumerated.)
LIMITED_VALUE_FIELDS = (
    "item_type",
    "category",
    "category_slug",
    "category_path",
    "dietary_tags",
    "allergens_contains",
    "allergens_may_contain",
    "is_gluten_free_listed",
    "portion_unit",
    "servings",
)

MENU_ITEM = "menu_item"


@dataclass(frozen=True)
class MenuItem:
    """One menu item as browsing shows it: enough to describe it and to draw its card."""

    id: str
    slug: str
    name: str
    description: str | None = None
    ingredients: tuple[str, ...] = ()
    price_gbp: float | None = None
    image: str | None = None


def _menu_item(row: Mapping[str, Any], name: str) -> MenuItem:
    description = row.get("description")
    ingredients = row.get("ingredients")
    price = row.get("price_gbp")
    image = row.get("image")
    return MenuItem(
        id=str(row.get("id") or ""),
        slug=str(row.get("slug") or ""),
        name=name,
        description=description.strip() or None if isinstance(description, str) else None,
        ingredients=tuple(i for i in ingredients if isinstance(i, str))
        if isinstance(ingredients, list)
        else (),
        price_gbp=float(price) if isinstance(price, int | float) else None,
        image=image if isinstance(image, str) and image else None,
    )


def _normalise(text: str) -> str:
    """Lower-cased words separated by single spaces, punctuation dropped: "Roku G+T?" and
    "roku g t" compare equal, so a guest's punctuation cannot hide a dish name."""
    return " ".join(re.sub(r"[\W_]+", " ", text.lower()).split())


@dataclass(frozen=True)
class MenuCatalog:
    """What the knowledge base contains and how it is organised."""

    total_rows: int = 0
    # item_type -> number of rows, e.g. {"menu_item": 162, "faq": 35}
    item_type_counts: dict[str, int] = field(default_factory=dict)
    # item_type -> group (category_path[0]) -> categories (category_path[-1]), all sorted
    type_tree: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    # menu items only: group -> category -> item names, all sorted. What browsing lists.
    menu_items: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    # the same items with their details (description, ingredients, price, image), same order
    menu_item_details: dict[str, dict[str, tuple[MenuItem, ...]]] = field(default_factory=dict)
    # limited-value field -> every value it takes, sorted (None is written "null")
    field_values: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.menu_items

    @property
    def groups(self) -> list[str]:
        """The menu's top-level groups (category_path[0] of menu items), sorted."""
        return sorted(self.menu_items)

    @property
    def categories(self) -> list[str]:
        """Every distinct category under any menu group, sorted."""
        return sorted({c for cats in self.menu_items.values() for c in cats})

    def categories_of(self, group: str) -> list[str]:
        return sorted(self.menu_items.get(group, {}))

    def items_of(self, group: str, category: str) -> list[str]:
        return list(self.menu_items.get(group, {}).get(category, ()))

    def item_details_of(self, group: str, category: str) -> list[MenuItem]:
        return list(self.menu_item_details.get(group, {}).get(category, ()))

    def groups_with_category(self, category: str) -> list[str]:
        """The groups a category appears under (more than one for e.g. `ramen`)."""
        return sorted(g for g, cats in self.menu_items.items() if category in cats)

    @cached_property
    def _item_names(self) -> frozenset[str]:
        names = {
            _normalise(name)
            for categories in self.menu_items.values()
            for items in categories.values()
            for name in items
        }
        return frozenset(name for name in names if name)

    def mentions_menu_item(self, text: str) -> bool:
        """True if `text` contains the full name of any menu item, as whole words and ignoring
        case and punctuation. Used to catch a dish question the model wrongly called off-topic."""
        padded = f" {_normalise(text)} "
        return any(f" {name} " in padded for name in self._item_names)


def _path(row: Mapping[str, Any]) -> list[str]:
    """The row's category_path as strings, falling back to [category] if it has none."""
    raw = row.get("category_path")
    path = [p for p in raw if isinstance(p, str)] if isinstance(raw, list) else []
    if not path and isinstance(row.get("category"), str):
        path = [row["category"]]
    return path


def _values(row: Mapping[str, Any], name: str) -> list[str]:
    """One row's contribution to a limited-value field, as strings."""
    value = row.get(name)
    if name == "category_path":
        return _path(row)
    if isinstance(value, list):
        return [str(v) for v in value]
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["true" if value else "false"]
    return [str(value)]


def build_catalog(rows: Iterable[Mapping[str, Any]]) -> MenuCatalog:
    """Build the catalog from corpus rows (property dicts). Pure: no I/O."""
    total = 0
    type_counts: Counter[str] = Counter()
    tree: dict[str, dict[str, set[str]]] = {}
    menu: dict[str, dict[str, list[str]]] = {}
    details: dict[str, dict[str, list[MenuItem]]] = {}
    values: dict[str, set[str]] = {name: set() for name in LIMITED_VALUE_FIELDS}

    for row in rows:
        total += 1
        raw_type = row.get("item_type")
        item_type = raw_type if isinstance(raw_type, str) else "unknown"
        type_counts[item_type] += 1
        for name in LIMITED_VALUE_FIELDS:
            values[name].update(_values(row, name))

        path = _path(row)
        if not path:
            continue
        group, category = path[0], path[-1]
        tree.setdefault(item_type, {}).setdefault(group, set()).add(category)
        item_name = row.get("name")
        if item_type == MENU_ITEM and isinstance(item_name, str) and item_name:
            menu.setdefault(group, {}).setdefault(category, []).append(item_name)
            details.setdefault(group, {}).setdefault(category, []).append(
                _menu_item(row, item_name)
            )

    def order(v: str) -> tuple[bool, str]:
        return (v == "null", v.lower())  # "null" last, otherwise case-insensitive

    return MenuCatalog(
        total_rows=total,
        item_type_counts=dict(sorted(type_counts.items())),
        type_tree={
            t: {g: tuple(sorted(cats)) for g, cats in sorted(groups.items())}
            for t, groups in sorted(tree.items())
        },
        menu_items={
            g: {c: tuple(sorted(names, key=str.lower)) for c, names in sorted(cats.items())}
            for g, cats in sorted(menu.items())
        },
        menu_item_details={
            g: {
                c: tuple(sorted(items, key=lambda item: item.name.lower()))
                for c, items in sorted(cats.items())
            }
            for g, cats in sorted(details.items())
        },
        field_values={name: tuple(sorted(vals, key=order)) for name, vals in values.items()},
    )
