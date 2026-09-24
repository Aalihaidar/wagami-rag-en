"""Item cards: which dishes an answer contains, and the card shown for each.

The rule (R-13 / C-21 in docs/LLM_RULES.md): a card is shown for every dish the answer under it
contains, in any way it does, and for no other dish. A card is never carried over from an earlier
answer, so everything here is a pure function of the current turn's final answer text and of what
this turn's search found.

`cards_for_answer()` resolves an answer to dishes in three layers, each adding to the result:

1. **The model's citation** (`cited_slugs`). The only layer that understands "it" and "the vegan
   one", so it is what catches an implicit reference whenever the model reports one. When the model
   cites nothing and the answer names nothing, there is no card: the search's best match is not
   guessed at.
2. **A full dish name in the answer**, among this turn's retrieved dishes only. Not among the menu
   at large: dish descriptions list components that are themselves menu items ("katsu curry
   sauce", "korean fried chicken"), so an answer describing what is in a dish would otherwise put
   a card on each component (measured on the real corpus: 31 of 162 dishes' name-free descriptions
   produced a card for some other dish that way, against 9 without it).
3. **A shortened name.** The start or the end of one retrieved dish's name, at least two words and
   at least 60% of them ("pork belly ramen" for "gochujang pork belly ramen (gluten-free recipe)"),
   that no other dish of this turn shares and that is not just one of the dish's own ingredients.
   Runs from the middle of a name, or shorter than that, are left out: measured on the real corpus
   this keeps about 80% of realistic shortenings and 3 false cards in 162, where any run of words
   gave 9.

Two rules limit those layers. A dish that is named only as a component of another named dish (its
own description or ingredients say "katsu curry sauce") gets no card unless the model cited it; the
live model answering "tell me about the chicken katsu curry" listed the sauce among the
ingredients. And a dish a filter screened out (the allergen or diet NOTE) never gets a card: it is
not among the retrieved dishes, and its name is also held back from shortened-name matching, so
"katsu curry" in a sentence about the screened-out "Katsu Curry" cannot put the card of a retrieved
"Katsu Curry (vegan recipe)" under it.

A name that only occurs inside a longer dish name that is also in the answer never counts on its
own (`coke` inside `diet coke`), see `catalog.mentioned_names()`.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

from app.agent.catalog import MenuItem, mentioned_names, normalise
from app.agent.choice_images import dish_image_path
from app.retrieval import MenuRow, RerankHit, plist, pnum, pstr


class CitedItem(TypedDict):
    id: str
    slug: str
    name: str
    description: str | None
    ingredients: list[str]
    price_gbp: float | None
    image: str


class ChoiceCard(TypedDict):
    """A card for one group or category in a list of them: its name and its cover picture (a
    file path, computed from the names by app.agent.choice_images -- not read from the corpus,
    so its existence is never checked here)."""

    name: str
    image: str


class Choices(TypedDict):
    """A reply that lists groups or categories, cut into the parts the chat page lays out: the
    opening sentence, one card per name, the closing question (rule R-14, guarantee C-23)."""

    intro: str
    outro: str
    cards: list[ChoiceCard]


def card_for_item(item: MenuItem) -> CitedItem | None:
    """The card for a catalog item, or None if it has no image (a card without a picture is not
    shown)."""
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


def card_for_row(row: MenuRow) -> CitedItem | None:
    """The card for a retrieved row, or None for an FAQ row or a dish with no image.

    The card is deliberately a glance-level summary: name, description, ingredients, and price
    only. Dietary tags, allergens, and nutrition are intentionally NOT carried through here --
    they're still real CONTEXT fields (format_row()) that the model sees and can state in the
    answer text itself (see GENERATION_SYSTEM_PROMPT's rule on this), just not duplicated onto
    the card. `name` also doubles as the image's `alt` text (Section B/4's accessibility
    requirement). `description` is `None` on the 17/162 corpus rows that genuinely have none
    (plain drinks, mostly); `ingredients` is a derived, not source-verified field -- both are
    omitted by the frontend rather than shown as a placeholder when empty.
    """
    image = pstr(row, "image")
    if pstr(row, "item_type") != "menu_item" or not image:
        return None
    return {
        "id": row["uuid"],
        "slug": pstr(row, "slug"),
        "name": pstr(row, "name"),
        "description": pstr(row, "description") or None,
        "ingredients": plist(row, "ingredients"),
        "price_gbp": pnum(row, "price_gbp"),
        "image": dish_image_path(
            [p for p in plist(row, "category_path") if isinstance(p, str)], image
        ),
    }


def cited_items_from_ranked(ranked: list[RerankHit], cited_slugs: list[str]) -> list[CitedItem]:
    """The cards for the retrieved menu items whose slug is in `cited_slugs`, in ranked order.

    A slug gets one card even if two rows carry it (the same dish listed under two menu sections
    shares its slug).
    """
    wanted = set(cited_slugs)
    items: list[CitedItem] = []
    for hit in ranked:
        card = card_for_row(hit["row"])
        if card is not None and card["slug"] in wanted:
            wanted.discard(card["slug"])
            items.append(card)
    return items


# A shortened name keeps at least this share of the name's words (see the module docstring).
_MIN_SHORTENED_SHARE = 0.6
# Words that say nothing about which dish is meant, so a run that starts or ends with one is not a
# shortened name: "with rice", "gluten free", "vegan recipe".
_FILLER_WORDS = frozenset(
    {"a", "an", "and", "the", "with", "of", "in", "on", "or", "for", "to"}
    | {"free", "gluten", "vegan", "vegetarian", "recipe"}
)
_PARENTHESIS = re.compile(r"\s*\([^)]*\)")


@dataclass(frozen=True)
class _Dish:
    card: CitedItem
    ingredients: tuple[str, ...] = ()

    @property
    def contents(self) -> str:
        """What the dish is made of, as its own description and ingredients say, normalised and
        padded so a whole name can be looked for in it."""
        return f" {normalise(' '.join([self.card['description'] or '', *self.ingredients]))} "


def _shortened_names(name: str, ingredients: Iterable[str]) -> set[str]:
    """The ways a guest might shorten a dish's name, normalised: its start or its end, keeping at
    least two words and _MIN_SHORTENED_SHARE of them. The recipe note in brackets is dropped first,
    and a run that is one of the dish's own ingredients is left out, since the answer saying it is
    describing what is in the dish, not naming it."""
    words = normalise(_PARENTHESIS.sub("", name)).split()
    own_ingredients = [f" {normalise(i)} " for i in ingredients]
    runs: set[str] = set()
    for start in range(len(words)):
        for end in range(start + 2, len(words) + 1):
            run = words[start:end]
            if start > 0 and end < len(words):
                continue  # from the middle of the name
            if run[0] in _FILLER_WORDS or run[-1] in _FILLER_WORDS:
                continue
            if len(run) < _MIN_SHORTENED_SHARE * len(words):
                continue
            phrase = " ".join(run)
            if not any(f" {phrase} " in ingredient for ingredient in own_ingredients):
                runs.add(phrase)
    return runs


def _retrieved_dishes(ranked: list[RerankHit]) -> dict[str, _Dish]:
    """This turn's retrieved dishes that can have a card, by slug, in ranked order."""
    dishes: dict[str, _Dish] = {}
    for hit in ranked:
        card = card_for_row(hit["row"])
        if card is not None and card["slug"] not in dishes:
            dishes[card["slug"]] = _Dish(card, tuple(plist(hit["row"], "ingredients")))
    return dishes


def _phrase_table(
    candidates: dict[str, _Dish], screened_out: str | None = None
) -> dict[str, set[str]]:
    """phrase -> the slugs it stands for, for this turn's candidate dishes: each one's full name,
    and its shortened names unless another candidate shares one or it equals any full name.

    `screened_out` (a dish a filter removed) is added with no slugs: it never produces a card, but
    a longer name in the answer suppresses a shortened name inside it, and it removes a shortened
    name equal to it."""
    full: dict[str, set[str]] = {}
    for slug, dish in candidates.items():
        full.setdefault(normalise(dish.card["name"]), set()).add(slug)
    if screened_out and normalise(screened_out):
        full.setdefault(normalise(screened_out), set())
    shortened: dict[str, set[str]] = {}
    for slug, dish in candidates.items():
        for phrase in _shortened_names(dish.card["name"], dish.ingredients):
            shortened.setdefault(phrase, set()).add(slug)
    unique = {p: slugs for p, slugs in shortened.items() if len(slugs) == 1 and p not in full}
    return {**unique, **full}


def cards_for_answer(
    answer: str,
    *,
    ranked: list[RerankHit],
    cited_slugs: list[str],
    screened_out: str | None = None,
) -> list[CitedItem]:
    """The cards to show under one answer: every dish it contains, and no others.

    `ranked` is this turn's retrieved rows (empty when nothing was answerable), `cited_slugs` the
    model's own citation, already limited to `ranked`, and `screened_out` the name of the dish a
    filter removed, when the answer had to explain why. See the module docstring for how each is
    used. Cards come in ranked order.
    """
    candidates = _retrieved_dishes(ranked)
    table = _phrase_table(candidates, screened_out)
    named = {slug for phrase in mentioned_names(answer, table) for slug in table[phrase]}
    cited = set(cited_slugs) & candidates.keys()
    # A dish the answer only names because it is a component of another dish it names ("...katsu
    # curry sauce, pickles" in the description of a katsu curry) is not what the answer is about,
    # unless the model cited it.
    named -= {
        slug
        for slug in named - cited
        if any(
            f" {normalise(candidates[slug].card['name'])} " in candidates[other].contents
            for other in named
            if other != slug
        )
    }
    found = named | cited
    return [dish.card for slug, dish in candidates.items() if slug in found]
