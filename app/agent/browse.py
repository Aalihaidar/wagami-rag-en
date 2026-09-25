"""Replies that need no search and no second model call.

Three kinds of message are answered straight from a fixed reply or from the MenuCatalog:
a greeting, an off-topic message, and browsing the menu ("what's on the menu?", "show me the
drinks", "cocktails"). The query-understanding call decides which one a message is (and which
group or category a browse names); this module turns that decision into the reply. Because the
lists come from the corpus, they can never name a category that does not exist, and because no
retrieval runs there is nothing to rank or hallucinate.

When a browse lists groups or categories, the reply also comes cut into an opening sentence, a
picture card per name -- the cover picture in that group's or category's own image folder (see
app.agent.choice_images) -- and a closing question, so the chat page can show cards where the
bullet list would be; the text with the list is kept for history.

When a browse reaches the items of a category, each item is described (description, ingredients,
price) and gets a card with its image, exactly as in a single-dish answer -- again from the
catalog, so it costs no search and no model call. When every item in the category has an image,
the reply is cut the same way as a groups/categories listing: an opening sentence, the item cards
(which already carry the description, ingredients and price the bullet line would have shown),
and a closing question, so the chat page shows the cards instead of the bullet list. An item with
no image still needs the bullet line to be seen at all, so when any item lacks one, the bullet
list is kept and shown alongside the cards, same as before.

Wording lives in app/agent/prompts.py; this module only decides what to say and fills the lists.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from app.agent.cards import ChoiceCard, Choices, CitedItem, card_for_item
from app.agent.catalog import MenuCatalog, MenuItem
from app.agent.choice_images import category_image_filename, group_image_filename
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
    """A direct reply: the text (whole, with any bullet list -- what is saved to history), the item
    cards to show under it (empty unless the reply lists a category's items), and, when the reply
    lists groups or categories, that reply cut around its list with a card per name.

    `intro`/`outro` are that same cut, but for a category's item listing (R-14/C-25): set only
    when every item in the listing got a card, so the chat page can show the cards in place of
    the bullet list instead of alongside it. None when an item had no image, or for any other
    kind of reply."""

    text: str
    cards: list[CitedItem] = field(default_factory=list)
    choices: Choices | None = None
    intro: str | None = None
    outro: str | None = None


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


# Stands in for {options} to find where a reply's list goes.
_LIST_MARK = "\x00options\x00"


def _choices(template: str, cards: list[ChoiceCard], **fields: str) -> DirectAnswer:
    """A fixed reply that lists `cards`' names: the whole text with the bullet list, and the same
    reply cut around the list. The wording stays the template's own."""
    text = template.format(options=_options([card["name"] for card in cards]), **fields)
    intro, outro = template.format(options=_LIST_MARK, **fields).split(_LIST_MARK)
    if not cards:
        return DirectAnswer(text)
    return DirectAnswer(
        text, choices={"intro": intro.strip(), "outro": outro.strip(), "cards": cards}
    )


def _overview(catalog: MenuCatalog) -> DirectAnswer:
    cards: list[ChoiceCard] = [
        {"name": group, "image": group_image_filename(group), "group": group, "category": None}
        for group in catalog.groups
    ]
    return _choices(MENU_OVERVIEW_REPLY, cards)


def _items(catalog: MenuCatalog, group: str, category: str) -> DirectAnswer:
    items = catalog.item_details_of(group, category)
    label = category if category == group else f"{category} ({group})"
    text = CATEGORY_ITEMS_REPLY.format(
        label=label, options="\n".join(_describe(item) for item in items)
    )
    cards = [card for item in items if (card := card_for_item(item)) is not None]
    if not cards or len(cards) < len(items):
        # An item with no image would otherwise vanish from the reply entirely, so keep the
        # bullet list as a fallback instead of cutting it away.
        return DirectAnswer(text, cards)
    intro, outro = CATEGORY_ITEMS_REPLY.format(label=label, options=_LIST_MARK).split(_LIST_MARK)
    return DirectAnswer(text, cards, intro=intro.strip(), outro=outro.strip())


def _group(catalog: MenuCatalog, group: str) -> DirectAnswer:
    """A group's categories -- or, when it has only one, that category's items, since a menu
    with a single choice is not worth a question."""
    categories = catalog.categories_of(group)
    if not categories:
        return _overview(catalog)
    if len(categories) == 1:
        return _items(catalog, group, categories[0])
    cards: list[ChoiceCard] = [
        {
            "name": category,
            "image": category_image_filename(group, category),
            "group": group,
            "category": category,
        }
        for category in categories
    ]
    return _choices(GROUP_REPLY, cards, group=group)


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


def is_known_pick(catalog: MenuCatalog, group: str, category: str | None) -> bool:
    """True if a clicked card's group (and category) exist in the catalog, so the click can be
    answered as that browse without an understanding call (rule R-15). Anything else -- a stale
    page after the menu changed, a hand-made request -- goes through understanding like typed
    text."""
    if group not in catalog.groups:
        return False
    return category is None or category in catalog.categories_of(group)


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
