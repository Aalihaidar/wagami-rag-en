"use strict";

/* Vanilla JS chat client -- no build step, no framework (Section 2's language decision).
 * Talks only to same-origin /session and /chat; renders every message as escaped text
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

/** One cited item (name/description/ingredients/price/image) as a card -- deliberately just
 * these four text fields (Section 4): dietary tags, allergens, and nutrition are real
 * CONTEXT fields the model can see and state in the answer text itself, not duplicated onto
 * the card. Every text field is set via textContent -- never innerHTML -- same
 * rendering-safety discipline as the message bubble itself, even though this data is
 * admin-controlled KB content, not the guest's own input. */
function buildItemCard(item) {
  const card = document.createElement("div");
  card.className = "item-card";

  const imageButton = document.createElement("button");
  imageButton.type = "button";
  imageButton.className = "item-card__image-button";
  imageButton.setAttribute("aria-label", `View larger image of ${item.name}`);

  const img = document.createElement("img");
  img.className = "item-card__image";
  img.src = item.image;
  img.alt = item.name;
  img.loading = "lazy";
  imageButton.appendChild(img);
  imageButton.addEventListener("click", () => openLightbox(item.image, item.name, imageButton));
  card.appendChild(imageButton);

  const body = document.createElement("div");
  body.className = "item-card__body";

  const name = document.createElement("p");
  name.className = "item-card__name";
  name.textContent = item.name;
  body.appendChild(name);

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

/** Appends one message bubble. `text` is always set via textContent -- never HTML. */
function appendMessage(role, text, { items = [], scroll = true } = {}) {
  const shouldScroll = scroll && isNearBottom();

  const bubble = document.createElement("div");
  bubble.className = `message message--${role}`;
  bubble.textContent = text;

  if (items.length > 0) {
    const gallery = document.createElement("div");
    gallery.className = "item-cards";
    for (const item of items) {
      gallery.appendChild(buildItemCard(item));
    }
    bubble.appendChild(gallery);
  }

  messageLog.appendChild(bubble);
  if (shouldScroll) scrollToBottom();
  return bubble;
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

async function sendMessage(text) {
  appendMessage("guest", text);
  showTypingIndicator();
  setBusy(true);

  inFlightController = new AbortController();
  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
      signal: inFlightController.signal,
    });
    const data = await response.json();
    hideTypingIndicator();

    if (!response.ok) {
      appendMessage("error", data.error || "Something went wrong. Please try again.");
      return;
    }
    appendMessage("assistant", data.answer, { items: data.cited_items || [] });
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
