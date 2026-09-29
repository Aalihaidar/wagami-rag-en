"""The Telegram bot: a second chat surface next to the web page, driven by the same agent.

Telegram posts each update (a guest's message or a tap on one of the bot's buttons) to
/telegram/webhook (app/main.py), which answers 200 at once and hands the update to
`TelegramBot.handle()` in the background. That runs the same chat turn as /chat, with the
guest's chat id as the session (`telegram:<chat id>`), and sends the reply back through the Bot
API, laid out like the chat page does (`layOutBody()` in app/static/js/chat.js):

- a list of groups or categories: its text, with a button per name (a tap browses into it, with
  no understanding call, like a card click on the page, rule R-15);
- a category's item listing: its opening sentence, the dishes' photos, then its closing question
  with a button per dish ("Tell me about <name>", like a dish card's click);
- one cited dish: the answer, then the dish's photo with its facts as the caption;
- several cited dishes: the answer with a button per dish, then their photos;
- anything else: the answer alone.

Replies are plain text, never parse_mode HTML/Markdown: the page shows the model's answer as
text too (textContent), and nothing model-derived is ever interpreted as markup.

A button's `callback_data` may hold only 64 bytes, too few for a group, category or dish name,
so it carries a short hash of what the button does and the full pick is kept in Redis under
that hash (`BUTTON_TTL_SECONDS`). A tap on a button older than that says so instead.

Only private chats are answered: in a group, one shared session and rate limit would cover every
member.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from redis import Redis

from app.cost_control import CONVERSATION_LIMIT_REPLY
from app.schemas import CHAT_MESSAGE_MAX_LENGTH, BrowsePick, ChatResponse, CitedItem

logger = logging.getLogger("app.telegram")

# httpx logs every request's URL at INFO, and a Bot API URL carries the bot token
# (https://api.telegram.org/bot<token>/sendMessage): with the root logger at INFO, the token
# would be written to the logs on every reply.
logging.getLogger("httpx").setLevel(logging.WARNING)

API_BASE_URL = "https://api.telegram.org"

WELCOME_MESSAGE = (
    "This is a demo restaurant chatbot built by ENG Ali Haidar to showcase a retrieval-grounded "
    "AI assistant. It runs on free-tier tools, so responses may be slower or less polished "
    "than a production deployment would be. Ask about menu items, prices, nutrition, "
    "allergens, or general FAQs.\n\n"
    "As this is just a demo, I can't place an order or help with a severe-allergy emergency. "
    "A real restaurant deployment could add both.\n\n"
    "Send /reset any time to start a new conversation."
)
RESET_REPLY = "Done, I've cleared our conversation. What would you like to know?"
CONVERSATION_LIMIT_TELEGRAM_REPLY = (
    "This conversation has gotten pretty long! Please send /reset to start a new one so I can "
    "keep giving you my full attention."
)
NOT_TEXT_REPLY = "I can only read text messages. Please type your question."
UNKNOWN_COMMAND_REPLY = "I don't know that command. Send /start for help or /reset to start over."
TOO_LONG_REPLY = (
    f"That message is a bit long for me. Please keep it under {CHAT_MESSAGE_MAX_LENGTH} characters."
)
RATE_LIMITED_REPLY = "You're sending messages a little fast. Please wait a minute and try again."
BUSY_REPLY = "I'm busy with other guests right now. Please try again in a moment."
EXPIRED_BUTTON_REPLY = "That button has expired. Please ask your question again."
CHAT_DISABLED_REPLY = "The chat is temporarily offline. Please try again later."

# Per-chat limit, the counterpart of /chat's per-IP one (CHAT_RATE_LIMIT in app/main.py): every
# update comes from Telegram's own servers, so a per-IP limit would put all guests in one bucket.
RATE_LIMIT_PER_MINUTE = 10
# Telegram retries an update it thinks went undelivered; one it has already sent is dropped.
UPDATE_SEEN_TTL_SECONDS = 24 * 60 * 60
BUTTON_TTL_SECONDS = 30 * 24 * 60 * 60

MESSAGE_MAX_LENGTH = 4096  # Bot API limit for a text message
CAPTION_MAX_LENGTH = 1024  # ... and for a photo's caption
MEDIA_GROUP_MAX_SIZE = 10  # ... and for the photos of one album

# What a turn needs from app/main.py: (session id, message, browse pick) -> the /chat reply.
RunTurn = Callable[[str, str, BrowsePick | None], ChatResponse]
ResetSession = Callable[[str], None]


@dataclass(frozen=True)
class Button:
    """One inline button: its label, and the chat turn a tap on it sends (the same message and
    browse pick as the matching card click on the page)."""

    label: str
    message: str
    browse: BrowsePick | None = None


@dataclass(frozen=True)
class Text:
    text: str
    buttons: list[Button] = field(default_factory=list)


@dataclass(frozen=True)
class Photos:
    """One or more photos, each with its own caption: a single photo or an album."""

    photos: list[tuple[str, str]]


Outgoing = Text | Photos


def session_id_for(chat_id: int) -> str:
    return f"telegram:{chat_id}"


def _price(item: CitedItem) -> str:
    return f" — £{item.price_gbp:.2f}" if item.price_gbp is not None else ""


def _short_caption(item: CitedItem) -> str:
    return f"{item.name}{_price(item)}"


def _detail_caption(item: CitedItem) -> str:
    """A single dish's facts, the text part of the page's detail view (rule C-26)."""
    lines = [_short_caption(item)]
    tags = [*item.dietary_tags, *(["gluten-free menu"] if item.is_gluten_free_listed else [])]
    if tags:
        lines.append(", ".join(tags))
    if item.description:
        lines.append(item.description)
    if (kcal := item.nutrition.get("kcal")) is not None:
        lines.append(f"{kcal:g} kcal per serving")
    if item.allergens_contains:
        lines.append(f"Contains: {', '.join(item.allergens_contains)}")
    if item.allergens_may_contain:
        lines.append(f"May contain: {', '.join(item.allergens_may_contain)}")
    if not (item.allergens_contains or item.allergens_may_contain):
        lines.append("No allergens declared.")
    return _truncate("\n".join(lines), CAPTION_MAX_LENGTH)


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _has_photo(item: CitedItem) -> bool:
    # The Bot API fetches a photo by URL itself, so only an absolute one works: with no
    # IMAGE_BASE_URL (or a relative one, e.g. local /images/menu) the dish goes without.
    return item.image.startswith(("https://", "http://"))


def _photos(items: list[CitedItem], caption: Callable[[CitedItem], str]) -> list[Outgoing]:
    shown = [(item.image, caption(item)) for item in items if _has_photo(item)]
    return [Photos(shown)] if shown else []


def _dish_buttons(items: list[CitedItem]) -> list[Button]:
    return [Button(item.name, f"Tell me about {item.name}") for item in items]


def layout(reply: ChatResponse) -> list[Outgoing]:
    """The messages one /chat reply becomes, in order -- see the module docstring."""
    items = reply.cited_items
    if reply.choices:
        buttons = [
            Button(
                card.name,
                f"Show me {card.category} from {card.group}"
                if card.category
                else f"Show me {card.group}",
                BrowsePick(group=card.group, category=card.category),
            )
            for card in reply.choices.cards
        ]
        text = f"{reply.choices.intro}\n\n{reply.choices.outro}"
        return [Text(text, buttons), *_photos(items, _short_caption)]
    if reply.intro is not None and reply.outro is not None:
        return [
            Text(reply.intro),
            *_photos(items, _short_caption),
            Text(reply.outro, _dish_buttons(items)),
        ]
    if len(items) == 1:
        item = items[0]
        if _has_photo(item):
            return [Text(reply.answer), Photos([(item.image, _detail_caption(item))])]
        return [Text(reply.answer), Text(_detail_caption(item))]
    return [Text(reply.answer, _dish_buttons(items)), *_photos(items, _short_caption)]


def split_text(text: str, limit: int = MESSAGE_MAX_LENGTH) -> list[str]:
    """`text` in pieces of at most `limit` characters, cut at a paragraph or line break (or a
    space) where there is one."""
    pieces = []
    while len(text) > limit:
        # The last paragraph break that fits, else the last line break, else the last space.
        cut = next(
            (pos for sep in ("\n\n", "\n", " ") if (pos := text.rfind(sep, 0, limit)) > 0),
            limit,
        )
        pieces.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text or not pieces:
        pieces.append(text)
    return pieces


class TelegramClient:
    """The few Bot API calls the bot makes, over one keep-alive HTTP pool. A failed call is
    logged (the method and Telegram's own reason -- never the URL, which holds the token) and
    returns None: a reply that can't be delivered must not break the rest of the update."""

    def __init__(self, token: str, *, transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(
            base_url=f"{API_BASE_URL}/bot{token}/",
            timeout=httpx.Timeout(15.0, connect=5.0),
            transport=transport,
        )

    def call(self, method: str, payload: dict[str, Any]) -> Any:
        try:
            body = self._http.post(method, json=payload).json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Telegram %s failed: %s", method, type(exc).__name__)
            return None
        if not body.get("ok"):
            logger.warning("Telegram %s rejected: %s", method, body.get("description"))
            return None
        return body.get("result")

    def close(self) -> None:
        self._http.close()


class TelegramBot:
    def __init__(
        self,
        client: TelegramClient,
        redis_client: Redis,
        run_turn: RunTurn,
        reset_session: ResetSession,
    ) -> None:
        self._client = client
        self._redis = redis_client
        self._run_turn = run_turn
        self._reset_session = reset_session

    # --- entry points -------------------------------------------------------------------

    def handle(self, update: dict[str, Any]) -> None:
        """Answer one update. Blocking (the Bot API calls and the chat turn), so app/main.py runs
        it in the threadpool."""
        update_id = update.get("update_id")
        if isinstance(update_id, int) and not self._first_delivery(update_id):
            return
        if isinstance(callback := update.get("callback_query"), dict):
            self._handle_button(callback)
        elif isinstance(message := update.get("message"), dict):
            self._handle_message(message)

    def reply_busy(self, update: dict[str, Any]) -> None:
        """Tell the guest the server is at its concurrency cap, without handling the update."""
        chat_id = private_chat_id(update)
        if chat_id is not None:
            self._send(chat_id, [Text(BUSY_REPLY)])

    # --- update kinds -------------------------------------------------------------------

    def _handle_message(self, message: dict[str, Any]) -> None:
        chat_id = _private_chat_id(message)
        if chat_id is None or not self._within_rate_limit(chat_id):
            return
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            self._send(chat_id, [Text(NOT_TEXT_REPLY)])
            return
        text = text.strip()
        if text.startswith("/"):
            self._handle_command(chat_id, text)
        elif len(text) > CHAT_MESSAGE_MAX_LENGTH:
            self._send(chat_id, [Text(TOO_LONG_REPLY)])
        else:
            self._answer(chat_id, text, None)

    def _handle_command(self, chat_id: int, text: str) -> None:
        # "/start", "/start <deep-link payload>" or "/start@wagami_restaurant_bot".
        command = text.split()[0].split("@")[0].lower()
        if command in ("/start", "/help"):
            self._send(chat_id, [Text(WELCOME_MESSAGE)])
        elif command == "/reset":
            self._reset_session(session_id_for(chat_id))
            self._send(chat_id, [Text(RESET_REPLY)])
        else:
            self._send(chat_id, [Text(UNKNOWN_COMMAND_REPLY)])

    def _handle_button(self, callback: dict[str, Any]) -> None:
        # Always answered, even when nothing else is done: until it is, the guest's app shows a
        # spinner on the button.
        self._client.call("answerCallbackQuery", {"callback_query_id": callback.get("id")})
        message = callback.get("message")
        chat_id = _private_chat_id(message) if isinstance(message, dict) else None
        if chat_id is None or not self._within_rate_limit(chat_id):
            return
        button = self._load_button(callback.get("data"))
        if button is None:
            self._send(chat_id, [Text(EXPIRED_BUTTON_REPLY)])
            return
        self._answer(chat_id, button.message, button.browse)

    def _answer(self, chat_id: int, message: str, browse: BrowsePick | None) -> None:
        self._client.call("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        reply = self._run_turn(session_id_for(chat_id), message, browse)
        if reply.answer == CONVERSATION_LIMIT_REPLY:
            # That reply points at the page's "New chat" button; here it is /reset.
            reply = reply.model_copy(update={"answer": CONVERSATION_LIMIT_TELEGRAM_REPLY})
        self._send(chat_id, layout(reply))

    # --- sending ------------------------------------------------------------------------

    def _send(self, chat_id: int, outgoing: list[Outgoing]) -> None:
        for part in outgoing:
            if isinstance(part, Text):
                self._send_text(chat_id, part)
            else:
                self._send_photos(chat_id, part)

    def _send_text(self, chat_id: int, part: Text) -> None:
        pieces = split_text(part.text)
        for i, piece in enumerate(pieces):
            payload: dict[str, Any] = {"chat_id": chat_id, "text": piece}
            if part.buttons and i == len(pieces) - 1:
                payload["reply_markup"] = {"inline_keyboard": self._keyboard(part.buttons)}
            self._client.call("sendMessage", payload)

    def _send_photos(self, chat_id: int, part: Photos) -> None:
        for start in range(0, len(part.photos), MEDIA_GROUP_MAX_SIZE):
            chunk = part.photos[start : start + MEDIA_GROUP_MAX_SIZE]
            if len(chunk) == 1:
                url, caption = chunk[0]
                sent = self._client.call(
                    "sendPhoto", {"chat_id": chat_id, "photo": url, "caption": caption}
                )
            else:
                media = [{"type": "photo", "media": url, "caption": c} for url, c in chunk]
                sent = self._client.call("sendMediaGroup", {"chat_id": chat_id, "media": media})
            if sent is None:
                # Telegram couldn't fetch a picture (or the call failed): the captions still
                # carry the dishes' names and facts, so send those as text instead.
                self._send_text(chat_id, Text("\n\n".join(caption for _, caption in chunk)))

    def _keyboard(self, buttons: list[Button]) -> list[list[dict[str, str]]]:
        return [
            [{"text": _truncate(button.label, 64), "callback_data": self._save_button(button)}]
            for button in buttons
        ]

    # --- Redis-backed state ---------------------------------------------------------------

    def _save_button(self, button: Button) -> str:
        """Keep what `button` does in Redis and return the short key its callback_data carries.
        The key is a hash of the pick itself, so re-sending the same button reuses (and
        refreshes) one entry instead of adding another."""
        value = json.dumps(
            {
                "message": button.message,
                "browse": button.browse.model_dump() if button.browse else None,
            },
            sort_keys=True,
        )
        key = hashlib.sha256(value.encode()).hexdigest()[:32]
        self._redis.set(f"telegram:button:{key}", value, ex=BUTTON_TTL_SECONDS)
        return key

    def _load_button(self, data: object) -> Button | None:
        if not isinstance(data, str):
            return None
        raw = self._redis.get(f"telegram:button:{data}")
        if raw is None:
            return None
        try:
            value = json.loads(raw)
            browse = BrowsePick(**value["browse"]) if value["browse"] else None
            return Button(label="", message=value["message"], browse=browse)
        except ValueError, KeyError, TypeError:
            return None

    def _first_delivery(self, update_id: int) -> bool:
        return bool(
            self._redis.set(f"telegram:update:{update_id}", 1, nx=True, ex=UPDATE_SEEN_TTL_SECONDS)
        )

    def _within_rate_limit(self, chat_id: int) -> bool:
        """Count this update against the chat's per-minute limit. The first update over the limit
        is told so; any more in the same minute are dropped silently."""
        key = f"telegram:rate:{chat_id}"
        count = int(self._redis.incr(key))
        if count == 1:
            self._redis.expire(key, 60)
        if count == RATE_LIMIT_PER_MINUTE + 1:
            self._send(chat_id, [Text(RATE_LIMITED_REPLY)])
        return count <= RATE_LIMIT_PER_MINUTE


def _private_chat_id(message: dict[str, Any]) -> int | None:
    chat = message.get("chat")
    if not isinstance(chat, dict) or chat.get("type") != "private":
        return None
    chat_id = chat.get("id")
    return chat_id if isinstance(chat_id, int) else None


def private_chat_id(update: dict[str, Any]) -> int | None:
    """The private chat an update belongs to, if any -- a message's own, or the chat of the
    message whose button was tapped."""
    message = update.get("message")
    if not isinstance(message, dict):
        callback = update.get("callback_query")
        message = callback.get("message") if isinstance(callback, dict) else None
    return _private_chat_id(message) if isinstance(message, dict) else None
