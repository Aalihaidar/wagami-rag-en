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

const INTRO_MESSAGE =
  "This is a demo restaurant chatbot built by ENG Ali Haidar to showcase a retrieval-grounded " +
  "AI assistant. It runs on free-tier tools, so responses may be slower or less polished " +
  "than a production deployment would be -- ask about menu items, prices, nutrition, " +
  "allergens, or general FAQs.\n\n" +
  "As this is just a demo, I can't place an order or help with a severe-allergy emergency -- " +
  "a real restaurant deployment could add both.";

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

function isNearBottom() {
  return (
    messageLog.scrollHeight - messageLog.scrollTop - messageLog.clientHeight <
    NEAR_BOTTOM_THRESHOLD_PX
  );
}

function scrollToBottom() {
  messageLog.scrollTo({ top: messageLog.scrollHeight, behavior: "smooth" });
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
 * `browse` for a group or category card. The question is not shown as a guest bubble -- the
 * reply just appears -- but the server still saves it to history, so a follow-up can refer to
 * it. Ignored while a reply is still in flight, like the composer. */
function buildAskButton(className, name, question, browse = null) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `${className} card-ask`;
  button.textContent = name;
  button.setAttribute("aria-label", question);
  button.addEventListener("click", () => {
    if (messageInput.disabled || !sessionId) return;
    sendMessage(question, { browse, showGuestBubble: false });
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

  if (item.price_gbp != null) {
    const price = document.createElement("p");
    price.className = "item-card__price";
    price.textContent = `£${item.price_gbp.toFixed(2)}`;
    body.appendChild(price);
  }

  card.appendChild(body);
  return card;
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

/** Puts a finished reply into a bubble: its text (or, for a list of groups or categories, the
 * opening sentence, the cards and the closing question), then any item cards. `answer` is
 * authoritative, so it replaces the streamed preview. Keeps the view pinned to the bottom only
 * if the guest hadn't scrolled away. */
function showReply(bubble, data) {
  const shouldScroll = isNearBottom();
  if (data.choices) {
    layOutChoices(bubble, data.choices);
  } else {
    bubble.textContent = data.answer;
  }
  appendItemCards(bubble, data.cited_items || []);
  if (shouldScroll) messageLog.scrollTop = messageLog.scrollHeight;
}

/** Appends one message bubble. `text` is always set via textContent -- never HTML. */
function appendMessage(role, text, { items = [], choices = null, scroll = true } = {}) {
  const shouldScroll = scroll && isNearBottom();

  const bubble = document.createElement("div");
  bubble.className = `message message--${role}`;
  if (choices) {
    layOutChoices(bubble, choices);
  } else {
    bubble.textContent = text;
  }
  appendItemCards(bubble, items);

  messageLog.appendChild(bubble);
  if (shouldScroll) scrollToBottom();
  return bubble;
}

/** Replaces a bubble's whole text (the streamed reply so far, or the final answer) and keeps the
 * view pinned to the bottom only if the guest hadn't scrolled away. Instant, not smooth: this
 * runs on every chunk, and queued smooth scrolls would lag behind the text. */
function setBubbleText(bubble, text) {
  const shouldScroll = isNearBottom();
  bubble.textContent = text;
  if (shouldScroll) messageLog.scrollTop = messageLog.scrollHeight;
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

/** Sends one guest message. `browse` is set only by a click on a group or category card, and
 * tells the server exactly which one (rule R-15). A card click also passes
 * `showGuestBubble: false`: `text` is sent and saved to history, but not shown in the chat. */
async function sendMessage(text, { browse = null, showGuestBubble = true } = {}) {
  if (showGuestBubble) appendMessage("guest", text);
  showTypingIndicator();
  // A card can be clicked from further up the log; with no guest bubble to follow, bring the
  // guest down to where the reply is about to appear.
  if (!showGuestBubble) scrollToBottom();
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
      if (event === "delta") {
        if (!bubble) {
          hideTypingIndicator();
          bubble = appendMessage("assistant", "");
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
          appendMessage("assistant", data.answer, {
            items: data.cited_items || [],
            choices: data.choices || null,
          });
        }
      } else if (event === "error") {
        finished = true;
        hideTypingIndicator();
        appendMessage("error", data.error || "Something went wrong. Please try again.");
      }
    }
    if (!finished) {
      hideTypingIndicator();
      appendMessage("error", "The reply was interrupted -- please try again.");
    }
  } catch (err) {
    hideTypingIndicator();
    if (err.name === "AbortError") {
      appendMessage("system", "Response cancelled.");
    } else {
      appendMessage(
        "error",
        "Couldn't reach the server -- please check your connection and try again."
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
