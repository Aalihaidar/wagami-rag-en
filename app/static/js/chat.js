"use strict";

/* Vanilla JS chat client -- no build step, no framework (Section 2's language decision).
 * Talks only to same-origin /session and /chat/stream (the reply arrives as Server-Sent
 * Events, read by sse.js); renders every message as escaped text
 * (textContent, never innerHTML of anything server- or model-derived) per Section 3's
 * client-side rendering-safety requirement.
 */

const SESSION_STORAGE_KEY = "chatSessionId";
const MAX_MESSAGE_LENGTH = 500;
const NEAR_BOTTOM_THRESHOLD_PX = 80;
// Space left above a new reply when the chat scrolls to its start.
const REPLY_TOP_GAP_PX = 12;

const INTRO_MESSAGE =
  "This is a demo restaurant chatbot built by ENG Ali Haidar to showcase a retrieval-grounded " +
  "AI assistant. It runs on free-tier tools, so responses may be slower or less polished " +
  "than a production deployment would be. Ask about menu items, prices, nutrition, " +
  "allergens, or general FAQs.\n\n" +
  "As this is just a demo, I can't place an order or help with a severe-allergy emergency. " +
  "A real restaurant deployment could add both.";

const messageLog = document.getElementById("message-log");
const composer = document.getElementById("composer");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const stopButton = document.getElementById("stop-button");
const newChatButton = document.getElementById("new-chat-button");
const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightbox-image");
const lightboxClose = document.getElementById("lightbox-close");

let sessionId = null;
let inFlightController = null;
let lightboxTrigger = null;
let stopFollowingReply = () => {};

// The page itself scrolls, not the message log: the log grows with the conversation and the
// composer area is sticky at the bottom of the window (chat.css's .page / .composer-area).
const pageScroller = document.scrollingElement || document.documentElement;

function isNearBottom() {
  return (
    pageScroller.scrollHeight - pageScroller.scrollTop - pageScroller.clientHeight <
    NEAR_BOTTOM_THRESHOLD_PX
  );
}

function scrollToBottom() {
  window.scrollTo({ top: pageScroller.scrollHeight, behavior: "smooth" });
}

// Keys that scroll the page when pressed outside the message box.
const SCROLL_KEYS = new Set(["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "]);

/** Scrolls the page so a new reply starts at the top of the window, and keeps doing so while the
 * reply grows (streamed text, cards, pictures loading) until its start has reached the top --
 * from there the guest reads down at their own pace, never pushed along by the text. A short
 * reply that cannot reach the top just leaves the page at its bottom. Stops at once if the guest
 * scrolls by hand, and when the next message is sent. */
function followReplyStart(bubble) {
  stopFollowingReply();
  const align = () => {
    const target = bubble.getBoundingClientRect().top + window.scrollY - REPLY_TOP_GAP_PX;
    // Instant, not smooth: this runs on every chunk, and queued smooth scrolls would lag.
    window.scrollTo({ top: target, behavior: "instant" });
    if (window.scrollY >= target - 1) stopFollowingReply();
  };
  const onKey = (event) => {
    if (SCROLL_KEYS.has(event.key) && event.target !== messageInput) stop();
  };
  const observer = new ResizeObserver(align);
  const stop = () => {
    observer.disconnect();
    window.removeEventListener("wheel", stop);
    window.removeEventListener("touchmove", stop);
    window.removeEventListener("keydown", onKey);
    stopFollowingReply = () => {};
  };
  stopFollowingReply = stop;
  window.addEventListener("wheel", stop, { passive: true });
  window.addEventListener("touchmove", stop, { passive: true });
  window.addEventListener("keydown", onKey);
  observer.observe(bubble);
}

const SVG_NS = "http://www.w3.org/2000/svg";

/** The small magnifier button on a card's picture: opens it in the zoom view. It sits above the
 * card's own click area (the name button's stretched ::after), so a click on it zooms instead of
 * asking about the card. */
function buildZoomButton(src, name) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "card-zoom";
  button.setAttribute("aria-label", `View larger image of ${name}`);
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(SVG_NS, "path");
  path.setAttribute(
    "d",
    "M10.5 4a6.5 6.5 0 1 0 4 11.6l4.4 4.4 1.4-1.4-4.4-4.4A6.5 6.5 0 0 0 10.5 4z" +
      "m0 2a4.5 4.5 0 1 1 0 9 4.5 4.5 0 0 1 0-9z"
  );
  svg.appendChild(path);
  button.appendChild(svg);
  button.addEventListener("click", () => openLightbox(src, name, button));
  return button;
}

/** A card's name as the button that asks about the card (rule R-15): its ::after stretches over
 * the whole card, so a click anywhere on it -- picture included -- sends `question`, with
 * `browse` for a group or category card. The chat shows only the card's name as the guest's
 * message, as if they had typed it; the full question is what the server answers and saves to
 * history, so a follow-up can refer to it. Ignored while a reply is still in flight, like the
 * composer. */
function buildAskButton(className, name, question, browse = null) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `${className} card-ask`;
  button.textContent = name;
  button.setAttribute("aria-label", question);
  button.addEventListener("click", () => {
    if (messageInput.disabled || !sessionId) return;
    sendMessage(question, { browse, displayText: name });
  });
  return button;
}

/** One cited item (name/description/ingredients/price/image) as a card -- deliberately just
 * these four text fields (Section 4): dietary tags, allergens, and nutrition are real
 * CONTEXT fields the model can see and state in the answer text itself, not duplicated onto
 * the card. Every text field is set via textContent -- never innerHTML -- same
 * rendering-safety discipline as the message bubble itself, even though this data is
 * admin-controlled KB content, not the guest's own input. */
function buildItemCard(item) {
  const card = document.createElement("div");
  card.className = "item-card";

  const frame = document.createElement("div");
  frame.className = "item-card__image-frame";
  const img = document.createElement("img");
  img.className = "item-card__image";
  img.src = item.image;
  img.alt = item.name;
  img.loading = "lazy";
  frame.appendChild(img);
  frame.appendChild(buildZoomButton(item.image, item.name));
  card.appendChild(frame);

  const body = document.createElement("div");
  body.className = "item-card__body";

  body.appendChild(buildAskButton("item-card__name", item.name, `Tell me about ${item.name}`));

  if (item.description) {
    const description = document.createElement("p");
    description.className = "item-card__description";
    description.textContent = item.description;
    body.appendChild(description);
  }

  if (item.ingredients && item.ingredients.length > 0) {
    const ingredients = document.createElement("p");
    ingredients.className = "item-card__ingredients";
    ingredients.textContent = `Ingredients: ${item.ingredients.join(", ")}`;
    body.appendChild(ingredients);
  }

  // Dietary tags and allergens as small pills (rule R-17): with them on the card, a multi-dish
  // answer need not list its dishes.
  const pills = el("div", "item-card__pills");
  const groups = [
    [item.dietary_tags, ""],
    [item.allergens_contains, "item-detail__tag--contains"],
    [item.allergens_may_contain, "item-detail__tag--may-contain"],
  ];
  for (const [values, tagClassName] of groups) {
    for (const value of values || []) {
      const pill = el("span", `item-detail__tag item-card__pill ${tagClassName}`.trim(), value);
      if (tagClassName === "item-detail__tag--may-contain") pill.title = "May contain";
      if (tagClassName === "item-detail__tag--contains") pill.title = "Contains";
      pills.appendChild(pill);
    }
  }
  if (pills.childElementCount > 0) body.appendChild(pills);

  if (item.price_gbp != null) {
    const price = document.createElement("p");
    price.className = "item-card__price";
    price.textContent = `£${item.price_gbp.toFixed(2)}`;
    body.appendChild(price);
  }

  card.appendChild(body);
  return card;
}

/** Lower-case words with punctuation dropped, so "Double Dutch Ginger-Beer" and "double dutch
 * ginger beer" compare equal -- the same comparison as the server's catalog.normalise(). */
function normaliseName(text) {
  return text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();
}

// A list line: "- ", "* ", "• ", "– " or "1. " / "1) " at its start.
const LIST_LINE = /^\s*(?:[-*•–]|\d+[.)])\s+/;

/** A multi-dish answer's text without its list of those dishes (rule R-17, C-27): each list
 * line that names one of the cards' dishes is dropped, since the card shows it. Everything else
 * stays. Display only -- the saved answer keeps the full text. */
function withoutListedDishes(text, items) {
  const names = items.map((item) => normaliseName(item.name)).filter(Boolean);
  const kept = text.split("\n").filter((line) => {
    if (!LIST_LINE.test(line)) return true;
    const words = ` ${normaliseName(line)} `;
    return !names.some((name) => words.includes(` ${name} `));
  });
  return kept
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** Appends the cited-item cards under a bubble's text. */
function appendItemCards(bubble, items) {
  if (items.length === 0) return;
  const gallery = document.createElement("div");
  gallery.className = "item-cards";
  for (const item of items) {
    gallery.appendChild(buildItemCard(item));
  }
  bubble.appendChild(gallery);
}

/** The nutrition figures of the single-dish view, in the card's `nutrition` order: the label
 * and unit each is shown with. */
const NUTRITION_LABELS = {
  kcal: ["Energy", "kcal"],
  protein_g: ["Protein", "g"],
  carbs_g: ["Carbs", "g"],
  sugars_g: ["Sugars", "g"],
  fat_g: ["Fat", "g"],
  sat_fat_g: ["Saturates", "g"],
  fibre_g: ["Fibre", "g"],
  sodium_g: ["Sodium", "g"],
  salt_g: ["Salt", "g"],
};

/** How a portion unit from the corpus reads on the page ("1 ea" -> "1 each"). */
const PORTION_UNITS = { ea: "each" };

/** A number as the corpus gives it, without float noise (15.3, not 15.299999). */
function formatNumber(value) {
  return value.toLocaleString("en-GB", { maximumFractionDigits: 2 });
}

/** An element with a class and, optionally, its text (always textContent, never HTML). */
function el(tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

/** One titled section of the single-dish panel, or null when it has nothing to show. */
function buildDetailSection(title, content) {
  if (!content) return null;
  const section = el("section", "item-detail__section");
  section.appendChild(el("h4", "item-detail__section-title", title));
  section.appendChild(content);
  return section;
}

/** A row of pill tags, or null for an empty list. `tagClassName` picks the colour. */
function buildPills(values, tagClassName = "") {
  if (!values || values.length === 0) return null;
  const list = el("div", "item-detail__pills");
  for (const value of values) {
    list.appendChild(el("span", `item-detail__tag ${tagClassName}`.trim(), value));
  }
  return list;
}

/** The per-serving nutrition as a grid of tiles, energy first; a figure the row has no value
 * for is left out. Null when there is none at all. */
function buildNutritionGrid(nutrition) {
  const grid = el("div", "item-detail__nutrition");
  for (const [field, [label, unit]] of Object.entries(NUTRITION_LABELS)) {
    const value = nutrition?.[field];
    if (value == null) continue;
    const tile = el("div", "item-detail__nutrient");
    if (field === "kcal") tile.classList.add("item-detail__nutrient--energy");
    const amount = el("span", "item-detail__nutrient-value", formatNumber(value));
    amount.appendChild(el("span", "item-detail__nutrient-unit", unit));
    tile.appendChild(amount);
    tile.appendChild(el("span", "item-detail__nutrient-label", label));
    grid.appendChild(tile);
  }
  return grid.childElementCount > 0 ? grid : null;
}

/** "Contains" and "May contain" as coloured pills, or a plain line when neither lists any. */
function buildAllergens(item) {
  const box = el("div", "item-detail__allergens");
  const groups = [
    ["Contains", item.allergens_contains, "item-detail__tag--contains"],
    ["May contain", item.allergens_may_contain, "item-detail__tag--may-contain"],
  ];
  for (const [label, allergens, tagClassName] of groups) {
    const pills = buildPills(allergens, tagClassName);
    if (!pills) continue;
    const row = el("div", "item-detail__allergen-row");
    row.appendChild(el("span", "item-detail__allergen-label", label));
    row.appendChild(pills);
    box.appendChild(row);
  }
  if (box.childElementCount === 0) {
    box.appendChild(el("p", "item-detail__note", "No allergens declared."));
  }
  return box;
}

/** The rest of the row as label/value pairs: where it sits on the menu, the gluten-free menu,
 * ABV for a drink, portion and servings. Null when there is nothing to list. */
function buildDetailList(item) {
  const rows = [];
  if (item.category_path && item.category_path.length > 0) {
    rows.push(["Menu", item.category_path.join(" › ")]);
  }
  rows.push(["Gluten-free menu", item.is_gluten_free_listed ? "Yes" : "No"]);
  if (item.abv_percent != null) {
    rows.push([
      "Alcohol",
      item.abv_percent > 0 ? `${formatNumber(item.abv_percent)}% ABV` : "Alcohol-free",
    ]);
  }
  if (item.portion_value != null) {
    const unit = PORTION_UNITS[item.portion_unit] ?? item.portion_unit ?? "";
    rows.push(["Portion", `${formatNumber(item.portion_value)} ${unit}`.trim()]);
  }
  if (item.servings) rows.push(["Serves", item.servings]);

  const list = el("dl", "item-detail__list");
  for (const [label, value] of rows) {
    const row = el("div", "item-detail__list-row");
    row.appendChild(el("dt", "item-detail__list-label", label));
    row.appendChild(el("dd", "item-detail__list-value", value));
    list.appendChild(row);
  }
  return list;
}

/** Lays out a single-dish answer (rules R-16, C-26), top to bottom: the assistant's written
 * answer (kept short by the prompt, G-11), a large picture (click to zoom, same as a card's),
 * then everything the dish's row holds for a guest -- name and price, dietary tags,
 * description, nutrition, allergens, ingredients and the remaining details -- as a structured
 * panel. Used only when the answer cites exactly one dish, not a listing (which gets its own
 * cut, layOutItemListing()) or a multi-dish comparison (which keeps the plain-text answer with a
 * gallery of small cards below it). Replaces whatever the bubble held. */
function layOutItemDetail(bubble, answer, item) {
  bubble.textContent = "";

  if (answer) bubble.appendChild(el("p", "item-detail__answer", answer));

  const frame = el("div", "item-detail__image-frame");
  const img = el("img", "item-detail__image");
  img.src = item.image;
  img.alt = item.name;
  img.loading = "lazy";
  frame.appendChild(img);
  frame.appendChild(buildZoomButton(item.image, item.name));
  bubble.appendChild(frame);

  const body = el("div", "item-detail__body");

  const header = el("div", "item-detail__header");
  header.appendChild(el("h3", "item-detail__name", item.name));
  if (item.price_gbp != null) {
    header.appendChild(el("span", "item-detail__price", `£${item.price_gbp.toFixed(2)}`));
  }
  body.appendChild(header);

  const badges = [...(item.dietary_tags || [])];
  if (item.is_gluten_free_listed) badges.push("gluten-free menu");
  const tags = buildPills(badges);
  if (tags) body.appendChild(tags);

  if (item.description) body.appendChild(el("p", "item-detail__description", item.description));

  const sections = [
    buildDetailSection("Nutrition per serving", buildNutritionGrid(item.nutrition)),
    buildDetailSection("Allergens", buildAllergens(item)),
    buildDetailSection("Ingredients", buildPills(item.ingredients, "item-detail__tag--plain")),
    buildDetailSection("Details", buildDetailList(item)),
  ];
  for (const section of sections) if (section) body.appendChild(section);

  bubble.appendChild(body);
}

/** One group or category in a list of them: its cover picture and its name. A click on the card
 * browses into it (rule R-15) -- a group's card shows its categories, a category's card its
 * items -- by sending the card's group/category with a readable question. A missing/failed
 * picture is dropped and the card is just its name. Every text field is set via textContent, as
 * for the item cards. */
function buildChoiceCard(card) {
  const item = document.createElement("li");
  item.className = "choice-card";

  if (card.image) {
    const frame = document.createElement("div");
    frame.className = "choice-card__picture-frame";
    const img = document.createElement("img");
    img.className = "choice-card__picture";
    img.src = card.image;
    img.alt = "";
    img.loading = "lazy";
    img.addEventListener("error", () => frame.remove());
    frame.appendChild(img);
    frame.appendChild(buildZoomButton(card.image, card.name));
    item.appendChild(frame);
  }

  const question = card.category
    ? `Show me ${card.category} from ${card.group}`
    : `Show me ${card.group}`;
  const browse = { group: card.group, category: card.category ?? null };
  item.appendChild(buildAskButton("choice-card__name", card.name, question, browse));
  return item;
}

/** Lays out a reply that lists groups or categories: the opening sentence, a card per name where
 * the bullet list would be, then the closing question. Replaces whatever the bubble held. */
function layOutChoices(bubble, choices) {
  bubble.textContent = "";

  const intro = document.createElement("div");
  intro.className = "message__part";
  intro.textContent = choices.intro;
  bubble.appendChild(intro);

  const list = document.createElement("ul");
  list.className = "choice-cards";
  for (const card of choices.cards) {
    list.appendChild(buildChoiceCard(card));
  }
  bubble.appendChild(list);

  const outro = document.createElement("div");
  outro.className = "message__part";
  outro.textContent = choices.outro;
  bubble.appendChild(outro);
}

/** Lays out a reply that lists a category's items: the opening sentence, an item card per dish
 * where the bullet list would be, then the closing question -- the same cut as layOutChoices(),
 * but with item cards (rule C-25). Replaces whatever the bubble held. */
function layOutItemListing(bubble, intro, outro, items) {
  bubble.textContent = "";

  const introEl = document.createElement("div");
  introEl.className = "message__part";
  introEl.textContent = intro;
  bubble.appendChild(introEl);

  appendItemCards(bubble, items);

  const outroEl = document.createElement("div");
  outroEl.className = "message__part";
  outroEl.textContent = outro;
  bubble.appendChild(outroEl);
}

/** Lays out a finished reply's body into `bubble`: a list of groups/categories as picture cards,
 * a category's item listing as item cards (both in place of the bullet list), a single cited
 * dish as the large image-and-facts detail view (rule C-26), or plain text with any item cards
 * appended below it (a comparison or recommendation citing several dishes at once). */
function layOutBody(bubble, { answer, cited_items: items = [], choices = null, intro, outro }) {
  if (choices) {
    layOutChoices(bubble, choices);
    appendItemCards(bubble, items);
  } else if (intro != null && outro != null) {
    layOutItemListing(bubble, intro, outro, items);
  } else if (items.length === 1) {
    layOutItemDetail(bubble, answer, items[0]);
  } else {
    bubble.textContent = items.length >= 2 ? withoutListedDishes(answer, items) : answer;
    appendItemCards(bubble, items);
  }
}

/** Puts a finished reply into a bubble. `answer` is authoritative, so it replaces the streamed
 * preview. Scrolling is left to followReplyStart(). */
function showReply(bubble, data) {
  layOutBody(bubble, data);
}

/** Appends one message bubble. `text` is always set via textContent -- never HTML. */
function appendMessage(
  role,
  text,
  { items = [], choices = null, intro = null, outro = null, scroll = true } = {}
) {
  const shouldScroll = scroll && isNearBottom();

  const bubble = document.createElement("div");
  bubble.className = `message message--${role}`;
  layOutBody(bubble, { answer: text, cited_items: items, choices, intro, outro });

  messageLog.appendChild(bubble);
  if (shouldScroll) scrollToBottom();
  return bubble;
}

/** Replaces a bubble's whole text (the streamed reply so far, or the final answer). Scrolling is
 * left to followReplyStart(), which keeps the reply's start in view rather than its end. */
function setBubbleText(bubble, text) {
  bubble.textContent = text;
}

function showTypingIndicator() {
  const indicator = document.createElement("div");
  indicator.className = "typing-indicator";
  indicator.id = "typing-indicator";
  indicator.setAttribute("role", "status");
  indicator.setAttribute("aria-label", "Assistant is typing");
  for (let i = 0; i < 3; i += 1) {
    const dot = document.createElement("span");
    dot.className = "typing-indicator__dot";
    indicator.appendChild(dot);
  }
  messageLog.appendChild(indicator);
  if (isNearBottom()) scrollToBottom();
}

function hideTypingIndicator() {
  document.getElementById("typing-indicator")?.remove();
}

function setComposerDisabled(disabled) {
  messageInput.disabled = disabled;
  sendButton.disabled = disabled;
}

function setBusy(isBusy) {
  setComposerDisabled(isBusy);
  // Stop takes Send's place rather than sitting beside it -- two buttons plus the input don't
  // fit on a narrow phone, and a disabled Send is dead space while a reply is in flight anyway.
  sendButton.hidden = isBusy;
  stopButton.hidden = !isBusy;
}

async function ensureSession() {
  const stored = sessionStorage.getItem(SESSION_STORAGE_KEY);
  if (stored) {
    sessionId = stored;
    return;
  }
  // Disabled for this gap too, not just during a chat send (setBusy) -- otherwise a guest
  // who submits while a fresh session id is still in flight would send session_id: null.
  setComposerDisabled(true);
  try {
    const response = await fetch("/session", { method: "POST" });
    const data = await response.json();
    sessionId = data.session_id;
    sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  } finally {
    setComposerDisabled(false);
  }
}

function renderIntro() {
  appendMessage("assistant", INTRO_MESSAGE, { scroll: false });
}

async function startNewChat() {
  const previousSessionId = sessionId;
  sessionStorage.removeItem(SESSION_STORAGE_KEY);
  stopFollowingReply();
  messageLog.textContent = "";
  sessionId = null;

  if (previousSessionId) {
    // Best-effort -- Redis's own TTL (Section 4) cleans up abandoned sessions regardless,
    // so a failed delete here (network hiccup, already-expired session) isn't fatal.
    fetch(`/session/${encodeURIComponent(previousSessionId)}`, { method: "DELETE" }).catch(
      () => {}
    );
  }

  renderIntro();
  await ensureSession();
  messageInput.focus();
}

/** How long a card click's reply waits behind the typing dots at least, so it follows the
 * guest's bubble instead of appearing with it. */
const CARD_REPLY_DELAY_MS = 700;

/** Resolves once `time` (a Date.now() value) has passed -- at once if it already has. */
function waitUntil(time) {
  const remaining = time - Date.now();
  if (remaining <= 0) return Promise.resolve();
  return new Promise((resolve) => setTimeout(resolve, remaining));
}

/** Sends one guest message. `browse` is set only by a click on a group or category card, and
 * tells the server exactly which one (rule R-15). A card click also passes `displayText`, the
 * card's name: that is what the guest bubble shows, while `text` is sent and saved to history. */
async function sendMessage(text, { browse = null, displayText = null } = {}) {
  stopFollowingReply();
  appendMessage("guest", displayText ?? text);
  showTypingIndicator();
  // Always, typed or clicked: the guest may be reading higher up (a card clicked from an earlier
  // reply, or the start of a long answer), and should see their message and the typing dots
  // at once, while the reply is being worked on -- not only once it arrives. When it does,
  // followReplyStart() takes over and brings its start to the top.
  scrollToBottom();
  // A card's reply is often ready at once (a browse needs no model call); holding it back for a
  // moment behind the typing dots lets the guest's bubble be seen first, then the answer.
  const replyNotBefore = displayText !== null ? Date.now() + CARD_REPLY_DELAY_MS : 0;
  setBusy(true);

  inFlightController = new AbortController();
  let bubble = null;
  let streamedText = "";
  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text, ...(browse && { browse }) }),
      signal: inFlightController.signal,
    });

    if (!response.ok) {
      // Rejections that happen before a stream starts (rate limit, busy, kill switch,
      // validation) are the usual JSON error body, not an event stream.
      const data = await response.json();
      hideTypingIndicator();
      appendMessage("error", data.error || "Something went wrong. Please try again.");
      return;
    }

    // Suppresses the log's screen-reader announcements until the reply is complete, so it isn't
    // read out again on every chunk.
    messageLog.setAttribute("aria-busy", "true");
    let finished = false;
    for await (const { event, data } of readSseEvents(response)) {
      if (!bubble) await waitUntil(replyNotBefore);
      if (event === "delta") {
        if (!bubble) {
          hideTypingIndicator();
          bubble = appendMessage("assistant", "", { scroll: false });
          followReplyStart(bubble);
        }
        streamedText += data.text;
        setBubbleText(bubble, streamedText.trimStart());
      } else if (event === "done") {
        finished = true;
        hideTypingIndicator();
        // `answer` is authoritative: it replaces the streamed preview, which differs from it
        // when the server swapped in a fallback reply mid-stream.
        if (bubble) {
          showReply(bubble, data);
        } else {
          const reply = appendMessage("assistant", data.answer, {
            items: data.cited_items || [],
            choices: data.choices || null,
            intro: data.intro || null,
            outro: data.outro || null,
            scroll: false,
          });
          followReplyStart(reply);
        }
      } else if (event === "error") {
        finished = true;
        hideTypingIndicator();
        appendMessage("error", data.error || "Something went wrong. Please try again.");
      }
    }
    if (!finished) {
      hideTypingIndicator();
      appendMessage("error", "The reply was interrupted. Please try again.");
    }
  } catch (err) {
    hideTypingIndicator();
    if (err.name === "AbortError") {
      appendMessage("system", "Response cancelled.");
    } else {
      appendMessage(
        "error",
        "Couldn't reach the server. Please check your connection and try again."
      );
    }
  } finally {
    messageLog.removeAttribute("aria-busy");
    inFlightController = null;
    setBusy(false);
    messageInput.focus();
  }
}

function openLightbox(src, alt, trigger) {
  lightboxTrigger = trigger;
  lightboxImage.src = src;
  lightboxImage.alt = alt;
  lightbox.hidden = false;
  lightboxClose.focus();
}

function closeLightbox() {
  lightbox.hidden = true;
  lightboxImage.src = "";
  lightboxTrigger?.focus();
  lightboxTrigger = null;
}

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value.trim().slice(0, MAX_MESSAGE_LENGTH);
  if (!text || messageInput.disabled) return;
  messageInput.value = "";
  sendMessage(text);
});

stopButton.addEventListener("click", () => {
  inFlightController?.abort();
});

newChatButton.addEventListener("click", () => {
  startNewChat();
});

lightboxClose.addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) closeLightbox();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !lightbox.hidden) closeLightbox();
});

(async function init() {
  renderIntro();
  await ensureSession();
  messageInput.focus();
})();
