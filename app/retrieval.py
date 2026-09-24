"""Weaviate hybrid-search retrieval tool for the KnowledgeBase collection.

Ported from the verified notebook source, `03_evaluation_groq.ipynb` -- see
docs/APP_AND_DEPLOYMENT_PLAN.md's LLM-provider note. Of the four candidate
`02_generation_checks_*` / `03_evaluation_*` notebooks, that is the only one
carrying all three retrieval-layer fixes this module depends on: the
`(gluten-free recipe)`/`(vegan recipe)` suffix being kept in `search_query`
(handled upstream, in app/agent/'s query-understanding prompt -- noted here
because it constrains what `search_query` in a QueryUnderstanding may
legitimately contain), the unfiltered-lookup fallback in `search()` below,
and `alcohol_free` filtering `abv_percent <= 0.5` rather than `IS NULL`.

This module makes no LLM calls itself -- query understanding happens
upstream in app/agent/ and is passed in here as a QueryUnderstanding dict,
so this module's logic can be unit-tested without a live LLM.
"""

import json
import math
import time
from typing import Any, TypedDict, cast

import httpx
import weaviate
from weaviate import WeaviateClient
from weaviate.classes.init import AdditionalConfig, Auth, Timeout
from weaviate.classes.query import Filter, FilterReturn, MetadataQuery
from weaviate.collections import Collection

from app.config import Settings
from app.timing import timed

FIELDS = [
    "name",
    "slug",
    "category",
    "category_path",
    "item_type",
    "description",
    "ingredients",
    "price_gbp",
    "kcal",
    "protein_g",
    # the rest of the nutrition and portion fields: not in the CONTEXT rows (format_row()), only
    # on the single-dish detail view's card (rule C-26)
    "carbs_g",
    "sugars_g",
    "fat_g",
    "sat_fat_g",
    "fibre_g",
    "sodium_g",
    "salt_g",
    "portion_value",
    "portion_unit",
    "servings",
    "abv_percent",
    "dietary_tags",
    "allergens_contains",
    "allergens_may_contain",
    "is_gluten_free_listed",
    "embedding_text",
    "image",
]

ALPHA = 0.75
K = 20
TOP_N = 6
GATE = 0.15
RERANK_MODEL = "rerank-v3.5"
COHERE_MAX_RPM = 15.0  # client-side pacing cap for rerank() -- edit to match your key's quota
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
COHERE_HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

RELAXABLE_FIELDS = [
    "kcal_max",
    "protein_min_g",
    "gluten_free_only",
    "alcohol_free",
    "price_max_gbp",
]


class QueryUnderstanding(TypedDict):
    """The retrieval-relevant fields of one query-understanding result.

    Produced upstream by app/agent/'s understand_query() (an LLM call). That
    function's own result type adds `intent` and `usage` on top of this via
    TypedDict inheritance.
    """

    dietary: str | None
    price_max_gbp: float | None
    allergens_exclude: list[str]
    search_query: str
    category_hint: list[str]
    gluten_free_only: bool
    kcal_max: float | None
    protein_min_g: float | None
    alcohol_free: bool


class MenuRow(TypedDict):
    """A plain, serializable snapshot of one Weaviate hit -- never the live SDK object.

    Converted immediately on retrieval (RetrievalTool.retrieve()) rather than passed through
    as a live weaviate.collections.classes.internal.Object: that type isn't msgpack-
    serializable, and app/agent/'s LangGraph checkpointer persists the full graph state
    (including SearchResult) after every node run, which would otherwise hard-crash the
    first time a session with a checkpointer actually ran a retrieval.
    """

    uuid: str
    score: float
    properties: dict[str, Any]


class RerankHit(TypedDict):
    row: MenuRow
    rerank: float
    hybrid: float


class ExcludedTopMatch(TypedDict):
    name: str
    reason: str


class SearchResult(TypedDict):
    relaxed_fields: list[str]
    excluded_top_match: ExcludedTopMatch | None
    search_text: str
    excluded: list[str]
    retrieved: int
    kept: int
    ranked: list[RerankHit]
    top: float
    answerable: bool
    rerank_search_units: int


def connect(settings: Settings) -> WeaviateClient:
    """Connect to Weaviate Cloud with the read-only key -- never the admin key `scripts/` use.

    Explicit timeouts (Section E), not the client's own defaults: a query timeout short
    enough that a stalled Weaviate call fails fast into app/main.py's outbound-error fallback
    reply instead of tying up a chat turn (and a free-tier instance's one concurrency slot)
    indefinitely. insert isn't relevant here (this app never writes), left at the default.
    """
    header_name = (
        "X-OpenAI-Api-Key" if settings.embedding_provider == "openai" else "X-Cohere-Api-Key"
    )
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.weaviate_url,
        auth_credentials=Auth.api_key(settings.weaviate_read_api_key),
        headers={header_name: settings.embedding_api_key},
        additional_config=AdditionalConfig(timeout=Timeout(init=5, query=15)),
    )


def pstr(row: MenuRow, key: str) -> str:
    """properties[key] as a string, or "" if it is not one."""
    v = row["properties"].get(key)
    return v if isinstance(v, str) else ""


def plist(row: MenuRow, key: str) -> list[Any]:
    """properties[key] as a list, or [] if it is not one."""
    v = row["properties"].get(key)
    return v if isinstance(v, list) else []


def pnum(row: MenuRow, key: str) -> float | None:
    """properties[key] as a float, or None if it is not numeric."""
    v = row["properties"].get(key)
    return float(v) if isinstance(v, int | float) else None


def pbool(row: MenuRow, key: str) -> bool:
    """properties[key] as a bool, or False if it is not one."""
    return bool(row["properties"].get(key))


def allergen_set(row: MenuRow) -> set[str]:
    """Union of a row's allergens_contains and allergens_may_contain."""
    return {*plist(row, "allergens_contains"), *plist(row, "allergens_may_contain")}


def build_filter(u: QueryUnderstanding) -> FilterReturn | None:
    """Turn a QueryUnderstanding into a Weaviate server-side filter."""
    clauses: list[FilterReturn] = []
    if u["dietary"]:
        clauses.append(Filter.by_property("dietary_tags").contains_any([u["dietary"]]))
    if u["price_max_gbp"] is not None:
        clauses.append(Filter.by_property("price_gbp").less_or_equal(float(u["price_max_gbp"])))
    if u["gluten_free_only"]:
        clauses.append(Filter.by_property("is_gluten_free_listed").equal(True))
    if u["kcal_max"] is not None:
        clauses.append(Filter.by_property("kcal").less_or_equal(float(u["kcal_max"])))
    if u["protein_min_g"] is not None:
        clauses.append(Filter.by_property("protein_g").greater_or_equal(float(u["protein_min_g"])))
    if u["alcohol_free"]:
        # 0.5% ABV is the standard UK low/no-alcohol labeling threshold. abv_percent is
        # populated on every drink row (0.0 for confirmed non-alcoholic, real ABV otherwise)
        # and left null on food rows -- a range filter naturally excludes null, so this
        # never needs an explicit IS NULL check.
        clauses.append(Filter.by_property("abv_percent").less_or_equal(0.5))
    return Filter.all_of(clauses) if clauses else None


def build_search_text(u: QueryUnderstanding) -> str:
    """Turn a QueryUnderstanding into the text used for hybrid retrieval + rerank.

    Uses search_query (dietary/price/allergy wording already stripped) instead of the raw
    question, so those tokens stop diluting the vector match once build_filter() has already
    handled them deterministically. category_hint is appended as a soft signal -- it's plain
    text, not a hard filter, so a wrong guess can only fail to help, never hide the right row.
    """
    text = u["search_query"]
    if u["category_hint"]:
        text = f"{text} ({', '.join(u['category_hint'])})"
    return text


def _is_constraint_set(u: QueryUnderstanding, field: str) -> bool:
    value = cast(dict[str, Any], u)[field]
    return bool(value) if field in ("gluten_free_only", "alcohol_free") else value is not None


def _clear_constraint(u: QueryUnderstanding, field: str) -> QueryUnderstanding:
    """A copy of u with one relaxable constraint cleared back to its unset value."""
    cleared = dict(u)
    cleared[field] = False if field in ("gluten_free_only", "alcohol_free") else None
    return cleared  # type: ignore[return-value]


class RetrievalTool:
    """Hybrid search + rerank against one KnowledgeBase collection.

    Holds the Weaviate collection handle and the Cohere rerank pacing state,
    so it's instantiated once (FastAPI lifespan) and reused across requests
    rather than recreated per call.
    """

    def __init__(
        self,
        kb: Collection,
        cohere_api_key: str,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._kb = kb
        self._cohere_api_key = cohere_api_key
        self._last_cohere_call_at = 0.0
        # Shared keep-alive pool for rerank calls instead of a fresh TCP+TLS handshake each
        # time (~110ms to api.cohere.com, measured); httpx.Client is thread-safe.
        self._http = http_client or httpx.Client(timeout=COHERE_HTTP_TIMEOUT)

    def close(self) -> None:
        self._http.close()

    def retrieve(
        self,
        query: str,
        *,
        k: int = K,
        alpha: float = ALPHA,
        filters: FilterReturn | None = None,
    ) -> list[MenuRow]:
        """Run one hybrid (keyword + vector) query against KnowledgeBase.

        Converts every hit to a plain MenuRow immediately -- see MenuRow's docstring for why
        the live SDK object never leaves this method.
        """
        with timed("weaviate"):
            objs = self._kb.query.hybrid(
                query=query,
                alpha=alpha,
                limit=k,
                filters=filters,
                return_properties=FIELDS,
                return_metadata=MetadataQuery(score=True),
            ).objects
        return [
            {
                "uuid": str(o.uuid),
                "score": o.metadata.score or 0.0,
                "properties": dict(o.properties),
            }
            for o in objs
        ]

    def _pace_cohere_call(self) -> None:
        """Block just long enough to keep rerank() under COHERE_MAX_RPM requests/minute."""
        wait = (60.0 / COHERE_MAX_RPM) - (time.monotonic() - self._last_cohere_call_at)
        if wait > 0:
            with timed("rerank_pace"):
                time.sleep(wait)
        self._last_cohere_call_at = time.monotonic()

    def rerank(
        self, query: str, rows: list[MenuRow], *, top_n: int = TOP_N
    ) -> tuple[list[RerankHit], int]:
        """Rerank rows against query with Cohere; fall back to hybrid order on rate limit.

        Paces itself under COHERE_MAX_RPM before every attempt -- constraint relaxation in
        search() can call this more than once per question. Returns (ranked_hits,
        search_units) -- search_units is Cohere's own meta.billed_units.search_units for this
        call (real billing data, not an estimate); 0 on the hybrid-order fallback, since no
        billable rerank call actually completed.
        """
        if not rows:
            return [], 0
        docs = [pstr(o, "embedding_text") or pstr(o, "name") for o in rows]
        payload = json.dumps(
            {
                "model": RERANK_MODEL,
                "query": query,
                "documents": docs,
                "top_n": min(top_n, len(docs)),
            }
        ).encode()
        results = None
        search_units = 0
        for attempt in range(4):
            self._pace_cohere_call()
            try:
                resp = self._http.post(
                    COHERE_RERANK_URL,
                    content=payload,
                    headers={
                        "Authorization": f"Bearer {self._cohere_api_key.strip()}",
                        "Content-Type": "application/json",
                        # Cohere sits behind infrastructure that blocks a bare library-default
                        # User-Agent -- see docs/APP_AND_DEPLOYMENT_PLAN.md / the Groq
                        # notebooks' own note on the same Cloudflare behavior.
                        "User-Agent": "wagami-rag-en/app.retrieval",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                results = data["results"]
                search_units = data.get("meta", {}).get("billed_units", {}).get("search_units", 0)
                break
            except httpx.HTTPStatusError as e:
                retryable = e.response.status_code == 429 and attempt < 3
                if retryable:
                    time.sleep(2 * 4**attempt)
                else:
                    break
        if results is None:
            hits: list[RerankHit] = [
                {"row": o, "rerank": math.nan, "hybrid": o["score"]} for o in rows[:top_n]
            ]
            return hits, 0
        hits = [
            {
                "row": rows[r["index"]],
                "rerank": float(r["relevance_score"]),
                "hybrid": rows[r["index"]]["score"],
            }
            for r in results
        ]
        return hits, search_units

    def search(self, understanding: QueryUnderstanding, *, gate: float = GATE) -> SearchResult:
        """Retrieve, exclude allergens, rerank, and gate against one QueryUnderstanding.

        If nothing clears the gate, progressively drops one RELAXABLE_FIELDS constraint at a
        time (least essential first) and retries, rather than declining outright when a
        closer, slightly off-spec match exists. dietary and allergens_exclude are never
        relaxed -- those are safety and preference guarantees, not refinements, and silently
        dropping either could put an unsafe or unwanted dish in front of a guest.
        """
        current = understanding
        relaxed_fields: list[str] = []
        total_search_units = 0
        while True:
            server_filter = build_filter(current)
            exclude = set(current["allergens_exclude"])
            search_text = build_search_text(current)
            objs = self.retrieve(search_text, filters=server_filter)
            kept = [o for o in objs if not (allergen_set(o) & exclude)] if exclude else objs
            with timed("rerank"):
                ranked, search_units = self.rerank(search_text, kept)
            total_search_units += search_units
            top = ranked[0]["rerank"] if ranked else 0.0
            answerable = bool(ranked) if math.isnan(top) else top >= gate
            if answerable:
                break
            next_field = next((f for f in RELAXABLE_FIELDS if _is_constraint_set(current, f)), None)
            if next_field is None:
                break
            relaxed_fields.append(next_field)
            current = _clear_constraint(current, next_field)

        # A guest naming a specific dish is most likely asking about the single best
        # name-match in the WHOLE corpus, filters aside. If a dietary hard-filter
        # (build_filter()) or the allergen exclude just kept that exact dish out of `ranked`
        # entirely, that has to reach generation explicitly -- otherwise nothing distinguishes
        # "the dish you asked about doesn't meet your requirement" from "here's a different
        # dish that happens to rank high", and the model can silently conflate the two. One
        # extra unfiltered lookup catches both exclusion paths (found live: allergens_exclude
        # conflated two chicken-katsu dishes; dietary="vegan" conflated a "(vegan recipe)"
        # variant with its actually-vegan base dish, since build_filter()'s dietary_tags
        # filter runs server-side before objs is ever built).
        excluded_top_match: ExcludedTopMatch | None = None
        if current["dietary"] or exclude:
            unfiltered = self.retrieve(search_text, k=1, filters=None)
            if unfiltered:
                top_obj = unfiltered[0]
                kept_names = {pstr(h["row"], "name") for h in ranked}
                if pstr(top_obj, "name") not in kept_names:
                    reasons = []
                    hit_allergens = allergen_set(top_obj) & exclude
                    if hit_allergens:
                        reasons.append(f"contains {', '.join(sorted(hit_allergens))}")
                    if current["dietary"] and current["dietary"] not in plist(
                        top_obj, "dietary_tags"
                    ):
                        reasons.append(f"is not tagged {current['dietary']}")
                    if reasons:
                        excluded_top_match = {
                            "name": pstr(top_obj, "name"),
                            "reason": "; ".join(reasons),
                        }

        return {
            "relaxed_fields": relaxed_fields,
            "excluded_top_match": excluded_top_match,
            "search_text": search_text,
            "excluded": sorted(exclude),
            "retrieved": len(objs),
            "kept": len(kept),
            "ranked": ranked,
            "top": top,
            "answerable": answerable,
            "rerank_search_units": total_search_units,
        }
