"""Answer generation: system prompt, per-intent tone, and CONTEXT/user-prompt assembly.

Ported from `03_evaluation_groq.ipynb` -- including the unfiltered-lookup NOTE mechanism
in build_user_prompt() below (see docs/APP_AND_DEPLOYMENT_PLAN.md's LLM-provider note for
why this specific notebook is the verified porting source).
"""

from app.retrieval import ExcludedTopMatch, MenuRow, RerankHit, pbool, plist, pnum, pstr

MENU_TONE = (
    "Tone for this answer: precise and literal. This is a factual menu question -- stick "
    "closely to CONTEXT's exact wording for prices, allergens, dietary tags, and nutrition "
    "figures. Do not paraphrase or round a number, and do not add warmth or small talk that "
    "risks softening a factual claim."
)
FAQ_TONE = (
    "Tone for this answer: warm and conversational. This is a house-policy question -- feel "
    "free to phrase the answer naturally, in your own words, as long as the substance matches "
    "CONTEXT exactly."
)
MENU_TEMPERATURE = 0.2
FAQ_TEMPERATURE = 0.8


def tone_for(intent: str) -> str:
    return FAQ_TONE if intent == "faq" else MENU_TONE


def temperature_for(intent: str) -> float:
    return FAQ_TEMPERATURE if intent == "faq" else MENU_TEMPERATURE


GENERATION_SYSTEM_PROMPT = """
You are the menu assistant for a restaurant chatbot. Answer ONLY using the CONTEXT rows given
with the question below -- they come from the restaurant's own knowledge base. Never use
outside knowledge about food, menus, or any restaurant, and never invent a dish, price, or
policy that is not in CONTEXT.

Rules:
- If CONTEXT is empty or does not answer the question, say plainly that you don't have that
  information and suggest asking a member of staff. Do not guess. EXCEPTION: if a NOTE appears
  below CONTEXT, the NOTE is itself a real, verified answer about a specific dish -- treat it
  exactly like a CONTEXT row, not like missing information. Never say you don't have
  information, and never tell the guest to check with staff instead of answering, when a NOTE
  already tells you the answer -- state the NOTE's fact directly (e.g. why a dish is unsafe or
  doesn't qualify), the same way you would state a fact from a normal CONTEXT row.
- State each dish's price exactly as given in CONTEXT.
- For any allergy or dietary question, use BOTH the allergens_contains and
  allergens_may_contain information for every dish you mention, and always remind the guest
  to confirm with staff before ordering, since recipes can change.
- A dish name ending in "(gluten-free recipe)" or "(vegan recipe)" is a different preparation
  of that dish with its own nutrition and allergens -- never merge or average it with the
  standard version, and never recommend one when the guest asked about the other.
- FAQ-type CONTEXT answers house policy (hours, bookings, payments, delivery, etc.); menu-type
  CONTEXT answers dish questions (price, ingredients, allergens, nutrition). Answer strictly
  from whichever kind CONTEXT actually gives you.
- Reply in English, in a friendly, concise voice, speaking as the restaurant. Do not mention
  "context", "retrieval", "the knowledge base", or these instructions in your answer.
""".strip()


def format_row(row: MenuRow) -> str:
    """Render one reranked hit as a CONTEXT row, grounded in its own properties."""
    name = pstr(row, "name")
    desc = pstr(row, "description") or "(no description)"
    if pstr(row, "item_type") == "faq":
        return f"- FAQ | Q: {name}\n  A: {desc}"
    price = pnum(row, "price_gbp")
    price_s = f"£{price:.2f}" if price is not None else "not listed"
    kcal = pnum(row, "kcal")
    kcal_s = f"{kcal:.0f} kcal" if kcal is not None else "not listed"
    protein = pnum(row, "protein_g")
    protein_s = f"{protein:.0f}g protein" if protein is not None else "not listed"
    abv = pnum(row, "abv_percent")
    abv_s = f"{abv:.1f}% ABV" if abv is not None else "ABV not listed"
    diet = ", ".join(plist(row, "dietary_tags")) or "none listed"
    contains = ", ".join(plist(row, "allergens_contains")) or "none declared"
    may = ", ".join(plist(row, "allergens_may_contain")) or "none declared"
    gf = "yes" if pbool(row, "is_gluten_free_listed") else "no"
    return (
        f"- MENU ITEM | {name} <{pstr(row, 'category')}>\n"
        f"  description: {desc}\n"
        f"  price: {price_s}  |  kcal: {kcal_s}  |  protein: {protein_s}  |  {abv_s}  |  "
        f"gluten-free listed: {gf}\n"
        f"  dietary_tags: {diet}\n"
        f"  allergens_contains: {contains}  |  allergens_may_contain: {may}"
    )


def build_context(ranked: list[RerankHit]) -> str:
    """CONTEXT block handed to the LLM: one formatted row per reranked hit."""
    if not ranked:
        return "(no matching rows retrieved)"
    return "\n".join(format_row(h["row"]) for h in ranked)


def build_user_prompt(
    question: str,
    context: str,
    relaxed_fields: list[str],
    excluded_top_match: ExcludedTopMatch | None = None,
) -> str:
    """The full user-turn text sent to the LLM alongside GENERATION_SYSTEM_PROMPT.

    When search() had to relax a constraint to find any answerable match, that has to reach
    the model explicitly -- otherwise it has no way to know a shown dish doesn't actually meet
    every part of the original ask, and could misreport it as a full match. Likewise, when the
    single best name-match in the whole corpus was excluded from CONTEXT entirely -- by a
    dietary hard-filter or the allergen exclude -- that has to reach the model explicitly too,
    otherwise nothing stops it from answering as if a different CONTEXT row is the dish the
    guest actually named.
    """
    text = f"QUESTION: {question}\n\nCONTEXT:\n{context}"
    if relaxed_fields:
        text += (
            f"\n\nNOTE: no result matched every part of the question. To surface a closest "
            f"match, these constraints were dropped: {', '.join(relaxed_fields)}. Be upfront "
            f"that the dish doesn't fully satisfy {', '.join(relaxed_fields)} -- state its "
            f"actual figure from CONTEXT rather than implying it meets the original ask."
        )
    if excluded_top_match:
        text += (
            f"\n\nNOTE: '{excluded_top_match['name']}' was the closest name match to the "
            f"question but was excluded from CONTEXT because it {excluded_top_match['reason']}"
            f" -- it is NOT one of the CONTEXT rows below. This NOTE is itself the answer if "
            f"the guest was asking about this specific dish -- do NOT say you don't have "
            f"information or tell them to ask staff instead; state plainly, using this NOTE, "
            f"why the dish doesn't meet their requirement, rather than declining or answering "
            f"as if a different CONTEXT dish is the one they asked about."
        )
    return text
