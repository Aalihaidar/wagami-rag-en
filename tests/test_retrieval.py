import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from app.retrieval import (
    MenuRow,
    QueryUnderstanding,
    RetrievalTool,
    _clear_constraint,
    _is_constraint_set,
    allergen_set,
    build_filter,
    build_search_text,
    plist,
    pnum,
    pstr,
)


def make_row(
    name: str,
    *,
    price_gbp: float | None = None,
    dietary_tags: list[str] | None = None,
    allergens_contains: list[str] | None = None,
    allergens_may_contain: list[str] | None = None,
    score: float = 0.5,
) -> MenuRow:
    """A plain MenuRow, for testing pstr/plist/pnum/allergen_set/search() logic directly."""
    return {
        "uuid": f"uuid-{name}",
        "score": score,
        "properties": {
            "name": name,
            "item_type": "menu_item",
            "price_gbp": price_gbp,
            "dietary_tags": dietary_tags or [],
            "allergens_contains": allergens_contains or [],
            "allergens_may_contain": allergens_may_contain or [],
            "embedding_text": name,
        },
    }


def make_raw_obj(row: MenuRow) -> Any:
    """The live-SDK-shaped fake FakeKB.query.hybrid() returns -- RetrievalTool.retrieve()
    converts this into a MenuRow like the one make_row() builds directly."""
    return SimpleNamespace(
        uuid=row["uuid"],
        properties=row["properties"],
        metadata=SimpleNamespace(score=row["score"]),
    )


def make_understanding(**overrides: Any) -> QueryUnderstanding:
    base: QueryUnderstanding = {
        "dietary": None,
        "price_max_gbp": None,
        "allergens_exclude": [],
        "search_query": "yasai cha han",
        "category_hint": [],
        "gluten_free_only": False,
        "kcal_max": None,
        "protein_min_g": None,
        "alcohol_free": False,
    }
    return cast(QueryUnderstanding, {**base, **overrides})


def test_pstr_plist_pnum_missing_or_wrong_type() -> None:
    row = make_row("ramen", price_gbp=9.5, dietary_tags=["vegan"])
    assert pstr(row, "name") == "ramen"
    assert pstr(row, "price_gbp") == ""  # not a string
    assert plist(row, "dietary_tags") == ["vegan"]
    assert plist(row, "name") == []  # not a list
    assert pnum(row, "price_gbp") == 9.5
    assert pnum(row, "name") is None  # not numeric


def test_allergen_set_is_union_of_both_lists() -> None:
    row = make_row(
        "prawn toast", allergens_contains=["crustaceans"], allergens_may_contain=["soya"]
    )
    assert allergen_set(row) == {"crustaceans", "soya"}


def test_build_filter_none_when_no_constraints() -> None:
    assert build_filter(make_understanding()) is None


def test_build_filter_alcohol_free_uses_threshold_not_is_null() -> None:
    f = build_filter(make_understanding(alcohol_free=True))
    assert f is not None
    # Regression guard for the fixed bug: alcohol_free must filter abv_percent <= 0.5, not
    # IS NULL (which matched every menu row). A single-clause Filter.all_of() returns the
    # clause itself (a weaviate-internal _FilterValue, not part of FilterReturn's public
    # shape) -- cast(Any, ...) to inspect it since that's exactly what's under test here.
    filter_value = cast(Any, f)
    assert filter_value.operator.value == "LessThanEqual"
    assert filter_value.value == 0.5
    assert filter_value.target == "abv_percent"


def test_build_search_text_keeps_recipe_suffix_verbatim() -> None:
    u = make_understanding(search_query="yasai cha han (vegan recipe)")
    assert build_search_text(u) == "yasai cha han (vegan recipe)"


def test_build_search_text_appends_category_hint() -> None:
    u = make_understanding(search_query="starter", category_hint=["bao buns", "gyoza"])
    assert build_search_text(u) == "starter (bao buns, gyoza)"


def test_constraint_set_and_clear() -> None:
    u = make_understanding(kcal_max=500, alcohol_free=True)
    assert _is_constraint_set(u, "kcal_max") is True
    assert _is_constraint_set(u, "protein_min_g") is False
    cleared = _clear_constraint(u, "kcal_max")
    assert cleared["kcal_max"] is None
    assert u["kcal_max"] == 500  # original untouched
    cleared_bool = _clear_constraint(u, "alcohol_free")
    assert cleared_bool["alcohol_free"] is False


class FakeKB:
    """Stands in for the Weaviate collection -- filtered vs unfiltered calls return
    different canned results, modeling server-side filtering excluding a row."""

    def __init__(self, filtered_objs: list[Any], unfiltered_objs: list[Any]) -> None:
        self.filtered_objs = filtered_objs
        self.unfiltered_objs = unfiltered_objs
        self.calls: list[dict[str, Any]] = []

    @property
    def query(self) -> Any:
        kb = self

        class _Query:
            def hybrid(self, *, filters: Any, limit: int, **kwargs: Any) -> Any:
                kb.calls.append({"filters": filters, "limit": limit})
                objs = kb.unfiltered_objs if filters is None else kb.filtered_objs
                return SimpleNamespace(objects=objs[:limit])

        return _Query()


def _no_rerank(
    query: str, rows: list[MenuRow], *, top_n: int = 6
) -> tuple[list[dict[str, Any]], int]:
    """Deterministic stand-in for Cohere rerank: hybrid order, real scores."""
    return [{"row": o, "rerank": o["score"], "hybrid": o["score"]} for o in rows[:top_n]], 0


def test_search_reports_excluded_top_match_on_allergen_conflation(monkeypatch: Any) -> None:
    """The unfiltered-lookup fix: an allergen exclude removes the guest's named dish from
    `ranked` entirely, and search() must surface that instead of silently answering about
    a different dish."""
    unsafe_dish = make_row("chicken katsu curry", allergens_contains=["cereals containing gluten"])
    other_dish = make_row("hot chicken katsu curry")
    kb = FakeKB(
        filtered_objs=[make_raw_obj(other_dish)], unfiltered_objs=[make_raw_obj(unsafe_dish)]
    )
    tool = RetrievalTool(kb=kb, cohere_api_key="")  # type: ignore[arg-type]
    monkeypatch.setattr(tool, "rerank", _no_rerank)

    understanding = make_understanding(allergens_exclude=["cereals containing gluten"])
    result = tool.search(understanding)

    assert result["excluded_top_match"] == {
        "name": "chicken katsu curry",
        "reason": "contains cereals containing gluten",
    }


def test_search_reports_excluded_top_match_on_dietary_conflation(monkeypatch: Any) -> None:
    """Same fix, dietary path: the guest names a "(vegan recipe)" dish that's actually
    tagged only vegetarian in the corpus (a real, deliberate corpus trap) -- the server-side
    dietary="vegan" filter correctly excludes it, and search() must say so rather than
    silently answering about the different vegan dish that filter did keep."""
    named_dish = make_row("yasai cha han (vegan recipe)", dietary_tags=["vegetarian"])
    other_vegan_dish = make_row("edamame", dietary_tags=["vegan"], score=0.6)
    kb = FakeKB(
        filtered_objs=[make_raw_obj(other_vegan_dish)], unfiltered_objs=[make_raw_obj(named_dish)]
    )
    tool = RetrievalTool(kb=kb, cohere_api_key="")  # type: ignore[arg-type]
    monkeypatch.setattr(tool, "rerank", _no_rerank)

    understanding = make_understanding(dietary="vegan", search_query="yasai cha han (vegan recipe)")
    result = tool.search(understanding)

    assert result["excluded_top_match"] == {
        "name": "yasai cha han (vegan recipe)",
        "reason": "is not tagged vegan",
    }


def test_search_no_excluded_top_match_when_named_dish_is_kept(monkeypatch: Any) -> None:
    dish = make_row("vegan ramen", dietary_tags=["vegan"], score=0.9)
    raw = make_raw_obj(dish)
    kb = FakeKB(filtered_objs=[raw], unfiltered_objs=[raw])
    tool = RetrievalTool(kb=kb, cohere_api_key="")  # type: ignore[arg-type]
    monkeypatch.setattr(tool, "rerank", _no_rerank)

    result = tool.search(make_understanding(dietary="vegan"))

    assert result["excluded_top_match"] is None
    assert result["answerable"] is True


def test_search_relaxes_constraints_when_nothing_clears_gate(monkeypatch: Any) -> None:
    below_gate = make_raw_obj(make_row("plain rice", score=0.05))
    kb = FakeKB(filtered_objs=[below_gate], unfiltered_objs=[below_gate])
    tool = RetrievalTool(kb=kb, cohere_api_key="")  # type: ignore[arg-type]
    monkeypatch.setattr(tool, "rerank", _no_rerank)

    understanding = make_understanding(kcal_max=100, protein_min_g=50)
    result = tool.search(understanding)

    assert result["relaxed_fields"] == ["kcal_max", "protein_min_g"]
    assert result["answerable"] is False


def _rerank_tool(handler: Callable[[httpx.Request], httpx.Response]) -> RetrievalTool:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return RetrievalTool(kb=cast(Any, None), cohere_api_key=" cohere-key ", http_client=client)


@pytest.fixture
def _no_real_sleeping(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Neither the pacing wait nor the 429 backoff may actually sleep in a test."""
    waits: list[float] = []
    monkeypatch.setattr("app.retrieval.time.sleep", waits.append)
    return waits


def test_rerank_posts_to_cohere_and_maps_results_back_to_rows(
    _no_real_sleeping: list[float],
) -> None:
    rows = [make_row("miso ramen", score=0.4), make_row("katsu curry", score=0.7)]
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.2},
                ],
                "meta": {"billed_units": {"search_units": 1}},
            },
        )

    hits, units = _rerank_tool(handler).rerank("curry", rows)

    assert [h["row"]["properties"]["name"] for h in hits] == ["katsu curry", "miso ramen"]
    assert hits[0]["rerank"] == 0.9
    assert hits[0]["hybrid"] == 0.7
    assert units == 1
    assert seen[0].headers["Authorization"] == "Bearer cohere-key"


def test_rerank_retries_a_429_then_succeeds(_no_real_sleeping: list[float]) -> None:
    responses = [
        httpx.Response(429, json={}),
        httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]}),
    ]

    hits, _ = _rerank_tool(lambda r: responses.pop(0)).rerank("ramen", [make_row("ramen")])

    assert hits[0]["rerank"] == 0.8
    assert 2 in _no_real_sleeping  # the first backoff step


def test_rerank_falls_back_to_hybrid_order_on_a_non_retryable_status(
    _no_real_sleeping: list[float],
) -> None:
    rows = [make_row("a", score=0.9), make_row("b", score=0.5)]

    hits, units = _rerank_tool(lambda r: httpx.Response(401, json={})).rerank("q", rows)

    assert [h["row"]["properties"]["name"] for h in hits] == ["a", "b"]
    assert all(math.isnan(h["rerank"]) for h in hits)
    assert units == 0


def test_rerank_lets_a_connection_failure_propagate(_no_real_sleeping: list[float]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    with pytest.raises(httpx.ConnectError):
        _rerank_tool(handler).rerank("q", [make_row("a")])


def test_rerank_skips_the_call_entirely_when_there_is_nothing_to_rank() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    assert _rerank_tool(handler).rerank("q", []) == ([], 0)
