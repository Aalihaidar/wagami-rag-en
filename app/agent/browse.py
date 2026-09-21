"""Replies that need no search and no second model call.

Three kinds of message are answered straight from a fixed reply or from the MenuCatalog:
a greeting, an off-topic message, and browsing the menu ("what's on the menu?", "show me the
drinks", "cocktails"). The query-understanding call decides which one a message is (and which
group or category a browse names); this module turns that decision into the reply. Because the
lists come from the corpus, they can never name a category that does not exist, and because no
retrieval runs there is nothing to rank or hallucinate.

When a browse reaches the items of a category, each item is described (description, ingredients,
price) and gets a card with its image, exactly as in a single-dish answer -- again from the
catalog, so it costs no search and no model call.

Wording lives in app/agent/prompts.py; this module only decides what to say and fills the lists.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.agent.catalog import MenuCatalog, MenuItem
from app.agent.generation import CitedItem
from app.agent.prompts import (
    AMBIGUOUS_CATEGORY_REPLY,
    CATEGORY_ITEMS_REPLY,
    GREETING_REPLY,
    GROUP_REPLY,
    MENU_OVERVIEW_REPLY,
    OFF_TOPIC_REPLY,
)

# Intents answered without retrieval or generation.
DIRECT_INTENTS = frozenset({"greeting", "off_topic", "menu_browse"})


@dataclass(frozen=True)
class DirectAnswer:
    """A direct reply: the text, and the item cards to show under it (empty unless the reply
    lists a category's items)."""

    text: str
    cards: list[CitedItem] = field(default_factory=list)


def _options(names: list[str]) -> str:
    return "\n".join(f"- {name}" for name in names)


def _describe(item: MenuItem) -> str:
    """One line for an item: its name, then whatever is known about it, e.g.
    "- iced latte: sweeten with cane syrup; ingredients: milk, coffee; price: £2.50."
    A name with nothing known about it is just "- name"."""
    details = []
    description = (item.description or "").rstrip(". ")
    if description:
        details.append(description)
    # The corpus derives `ingredients` by splitting the description into fragments, so each
    # sentence of the description reappears as an "ingredient"; say it once. Only an exact
    # fragment is dropped, so a real ingredient the description merely mentions ("milk") stays.
    fragments = {part.strip(" .").lower() for part in re.split(r"\.\s+", description)}
    ingredients = [i for i in item.ingredients if i.strip(" .").lower() not in fragments]
    if ingredients:
        details.append("ingredients: " + ", ".join(ingredients))
    if item.price_gbp is not None:
        details.append(f"price: £{item.price_gbp:.2f}")
    return f"- {item.name}: {'; '.join(details)}." if details else f"- {item.name}"


def _card(item: MenuItem) -> CitedItem | None:
    """The card for an item, or None if it has no image (a card without a picture is not shown)."""
    if not item.image:
        return None
    return {
        "id": item.id,
        "slug": item.slug,
        "name": item.name,
        "description": item.description,
        "ingredients": list(item.ingredients),
        "price_gbp": item.price_gbp,
        "image": item.image,
    }


def _overview(catalog: MenuCatalog) -> DirectAnswer:
    return DirectAnswer(MENU_OVERVIEW_REPLY.format(options=_options(catalog.groups)))


def _items(catalog: MenuCatalog, group: str, category: str) -> DirectAnswer:
    items = catalog.item_details_of(group, category)
    label = category if category == group else f"{category} ({group})"
    text = CATEGORY_ITEMS_REPLY.format(
        label=label, options="\n".join(_describe(item) for item in items)
    )
    return DirectAnswer(text, [card for item in items if (card := _card(item)) is not None])


def _group(catalog: MenuCatalog, group: str) -> DirectAnswer:
    """A group's categories -- or, when it has only one, that category's items, since a menu
    with a single choice is not worth a question."""
    categories = catalog.categories_of(group)
    if not categories:
        return _overview(catalog)
    if len(categories) == 1:
        return _items(catalog, group, categories[0])
    return DirectAnswer(GROUP_REPLY.format(group=group, options=_options(categories)))


def _category(catalog: MenuCatalog, group: str | None, category: str) -> DirectAnswer:
    groups = catalog.groups_with_category(category)
    if group is not None and group in groups:
        return _items(catalog, group, category)
    # A bare name that is also a group ("drinks", "sides", "the main event") means the group:
    # that is what a guest who says it wants to see. Its category of the same name, where there
    # is one (a group's own category, or the kids' "drinks"), is reached by naming the group too.
    if category in catalog.menu_items:
        return _group(catalog, category)
    if len(groups) == 1:
        return _items(catalog, groups[0], category)
    if groups:  # e.g. ramen, which the kids' menu and the main menu both have
        return DirectAnswer(
            AMBIGUOUS_CATEGORY_REPLY.format(category=category, groups=" and ".join(groups))
        )
    return _overview(catalog)


def browse_answer(catalog: MenuCatalog, group: str | None, category: str | None) -> DirectAnswer:
    """The menu-browsing reply for a group and/or category the guest named (None if unnamed):

    * neither named -> the menu's groups, and ask which;
    * a group -> its categories (or its items, if it has just one category);
    * a category -> all its items with their details and cards, or which group is meant if the
      name is ambiguous.

    A name that is not in the catalog falls back to the overview instead of guessing.
    """
    if category:
        return _category(catalog, group, category)
    if group:
        return _group(catalog, group)
    return _overview(catalog)


def direct_answer(understanding: Mapping[str, Any], catalog: MenuCatalog) -> DirectAnswer:
    """The reply (and any item cards) for an understanding whose intent is in DIRECT_INTENTS."""
    intent = understanding["intent"]
    if intent == "greeting":
        return DirectAnswer(GREETING_REPLY)
    if intent == "off_topic":
        return DirectAnswer(OFF_TOPIC_REPLY)
    return browse_answer(
        catalog, understanding.get("browse_group"), understanding.get("browse_category")
    )


def browse_reply(catalog: MenuCatalog, group: str | None, category: str | None) -> str:
    """Just the text of browse_answer()."""
    return browse_answer(catalog, group, category).text


def direct_reply(understanding: Mapping[str, Any], catalog: MenuCatalog) -> str:
    """Just the text of direct_answer()."""
    return direct_answer(understanding, catalog).text
