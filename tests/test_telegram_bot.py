import io
import json
import logging
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from app.config import Settings
from app.cost_control import CONVERSATION_LIMIT_REPLY
from app.main import app
from app.schemas import BrowsePick, ChatResponse, ChoiceCard, Choices, CitedItem
from app.telegram_bot import (
    BUSY_REPLY,
    CONVERSATION_LIMIT_TELEGRAM_REPLY,
    EXPIRED_BUTTON_REPLY,
    NOT_TEXT_REPLY,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMITED_REPLY,
    RESET_REPLY,
    TOO_LONG_REPLY,
    WELCOME_MESSAGE,
    Grid,
    Photos,
    TelegramBot,
    TelegramClient,
    Text,
    layout,
    split_text,
)
from app.telegram_grid import build_grid

CHAT_ID = 42
SECRET = "test-webhook-secret"


class FakeRedis:
    """Just the commands the bot uses: set (with nx/ex), get, incr and expire."""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def set(self, key: str, value: Any, *, nx: bool = False, ex: int | None = None) -> bool:
        if nx and key in self.store:
            return False
        self.store[key] = value if isinstance(value, str) else str(value)
        return True

    def get(self, key: str) -> Any:
        value = self.store.get(key)
        return value.encode() if isinstance(value, str) else value

    def incr(self, key: str) -> int:
        self.store[key] = int(self.store.get(key, 0)) + 1
        return self.store[key]

    def expire(self, key: str, ttl: int) -> bool:
        return True


class FakeTelegramClient:
    """Records every Bot API call. `failing` names methods that fail ("upload" for any
    upload); `covers` maps a picture URL to the bytes `fetch()` returns for it (None otherwise)."""

    def __init__(
        self, *, failing: set[str] | None = None, covers: dict[str, bytes] | None = None
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.uploads: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        self.fetched: list[str] = []
        self._failing = failing or set()
        self._covers = covers or {}

    def call(self, method: str, payload: dict[str, Any]) -> Any:
        self.calls.append((method, payload))
        return None if method in self._failing else {"message_id": len(self.calls)}

    def upload(self, method: str, fields: dict[str, Any], files: dict[str, Any]) -> Any:
        self.uploads.append((method, fields, files))
        if "upload" in self._failing:
            return None
        return {"message_id": 99, "photo": [{"file_id": "small"}, {"file_id": "grid-file-id"}]}

    def fetch(self, url: str) -> bytes | None:
        self.fetched.append(url)
        return self._covers.get(url)

    def close(self) -> None:
        pass

    def sent(self, method: str = "sendMessage") -> list[dict[str, Any]]:
        return [payload for m, payload in self.calls if m == method]

    def texts(self) -> list[str]:
        return [payload["text"] for payload in self.sent()]


def dish(name: str, image: str = "https://img.example/menu/x.png", **extra: Any) -> CitedItem:
    return CitedItem(
        id=name,
        slug=name.lower().replace(" ", "-"),
        name=name,
        price_gbp=9.5,
        image=image,
        **extra,
    )


def reply(answer: str = "An answer.", **extra: Any) -> ChatResponse:
    return ChatResponse(session_id="s", answer=answer, cited_items=extra.pop("cited", []), **extra)


class Harness:
    def __init__(
        self,
        *,
        turn_reply: ChatResponse | None = None,
        failing: set[str] | None = None,
        covers: dict[str, bytes] | None = None,
    ):
        self.client = FakeTelegramClient(failing=failing, covers=covers)
        self.redis = FakeRedis()
        self.turns: list[tuple[str, str, BrowsePick | None]] = []
        self.resets: list[str] = []
        self.turn_reply = turn_reply or reply()
        self.bot = TelegramBot(
            self.client,  # type: ignore[arg-type]
            self.redis,  # type: ignore[arg-type]
            self.run_turn,
            self.resets.append,
        )
        self._next_update_id = 1

    def run_turn(self, session_id: str, message: str, browse: BrowsePick | None) -> ChatResponse:
        self.turns.append((session_id, message, browse))
        return self.turn_reply

    def send(self, text: str | None, *, chat_type: str = "private") -> None:
        message: dict[str, Any] = {"chat": {"id": CHAT_ID, "type": chat_type}}
        if text is not None:
            message["text"] = text
        self.bot.handle({"update_id": self._take_id(), "message": message})

    def tap(self, data: str) -> None:
        callback = {
            "id": "cb1",
            "data": data,
            "message": {"chat": {"id": CHAT_ID, "type": "private"}},
        }
        self.bot.handle({"update_id": self._take_id(), "callback_query": callback})

    def _take_id(self) -> int:
        self._next_update_id += 1
        return self._next_update_id


# --- layout ---------------------------------------------------------------------------------


def test_layout_plain_answer_is_one_text() -> None:
    assert layout(reply("Hello!")) == [Text("Hello!")]


def test_layout_choices_become_buttons_that_browse() -> None:
    choices = Choices(
        intro="Here are our menus:",
        outro="Which one?",
        cards=[
            ChoiceCard(name="drinks", image="c.png", group="drinks"),
            ChoiceCard(name="ramen", image="c.png", group="the main event", category="ramen"),
        ],
    )
    [text] = layout(reply(choices=choices))
    assert isinstance(text, Text)
    assert text.text == "Here are our menus:\n\nWhich one?"
    assert [(b.label, b.message) for b in text.buttons] == [
        ("drinks", "Show me drinks"),
        ("ramen", "Show me ramen from the main event"),
    ]
    assert text.buttons[1].browse == BrowsePick(group="the main event", category="ramen")


def test_layout_item_listing_is_intro_photos_then_outro_with_dish_buttons() -> None:
    items = [dish("Vegan Ramen"), dish("Chicken Ramen")]
    parts = layout(reply(cited=items, intro="Our ramen:", outro="Want details?"))
    assert parts[0] == Text("Our ramen:")
    assert parts[1] == Photos(
        [(items[0].image, "Vegan Ramen — £9.50"), (items[1].image, "Chicken Ramen — £9.50")]
    )
    outro = parts[2]
    assert isinstance(outro, Text)
    assert outro.text == "Want details?"
    assert [b.message for b in outro.buttons] == [
        "Tell me about Vegan Ramen",
        "Tell me about Chicken Ramen",
    ]


def test_layout_single_dish_gets_its_facts_as_the_photo_caption() -> None:
    item = dish(
        "Vegan Ramen",
        description="Miso broth.",
        dietary_tags=["vegan"],
        allergens_contains=["soya"],
        nutrition={"kcal": 512.0},
    )
    [text, photos] = layout(reply("It's lovely.", cited=[item]))
    assert text == Text("It's lovely.")
    assert isinstance(photos, Photos)
    caption = photos.photos[0][1]
    assert caption.splitlines() == [
        "Vegan Ramen — £9.50",
        "vegan",
        "Miso broth.",
        "512 kcal per serving",
        "Contains: soya",
    ]


def test_layout_without_absolute_image_urls_sends_no_photos() -> None:
    items = [dish("A", image="a.png"), dish("B", image="/images/menu/b.png")]
    parts = layout(reply("Two dishes.", cited=items))
    assert len(parts) == 1 and isinstance(parts[0], Text)
    [single_text, facts] = layout(reply("One dish.", cited=[items[0]]))
    assert isinstance(facts, Text) and facts.text.startswith("A — £9.50")


def test_split_text_cuts_at_line_breaks_and_keeps_everything() -> None:
    text = "\n".join(f"line {i}" for i in range(50))
    pieces = split_text(text, limit=60)
    assert all(len(piece) <= 60 for piece in pieces)
    assert "\n".join(pieces) == text
    assert split_text("short") == ["short"]


# --- the bot --------------------------------------------------------------------------------


def test_text_message_runs_a_turn_in_the_chats_own_session() -> None:
    harness = Harness(turn_reply=reply("Our ramen is great."))
    harness.send("  what ramen do you have?  ")
    assert harness.turns == [("telegram:42", "what ramen do you have?", None)]
    assert harness.client.sent("sendChatAction")[0]["action"] == "typing"
    assert harness.client.texts() == ["Our ramen is great."]


def test_start_and_reset_commands() -> None:
    harness = Harness()
    harness.send("/start")
    harness.send("/reset@wagami_restaurant_bot")
    assert harness.client.texts() == [WELCOME_MESSAGE, RESET_REPLY]
    assert harness.resets == ["telegram:42"]
    assert harness.turns == []


def test_non_text_and_too_long_messages_never_reach_the_agent() -> None:
    harness = Harness()
    harness.send(None)
    harness.send("x" * 501)
    assert harness.client.texts() == [NOT_TEXT_REPLY, TOO_LONG_REPLY]
    assert harness.turns == []


def test_group_chats_are_ignored() -> None:
    harness = Harness()
    harness.send("hello", chat_type="group")
    assert harness.client.calls == [] and harness.turns == []


def test_a_redelivered_update_is_handled_once() -> None:
    harness = Harness()
    update = {"update_id": 7, "message": {"chat": {"id": CHAT_ID, "type": "private"}, "text": "hi"}}
    harness.bot.handle(update)
    harness.bot.handle(update)
    assert len(harness.turns) == 1


def test_rate_limit_warns_once_then_drops() -> None:
    harness = Harness()
    for _ in range(RATE_LIMIT_PER_MINUTE + 3):
        harness.send("hi")
    assert len(harness.turns) == RATE_LIMIT_PER_MINUTE
    assert harness.client.texts().count(RATE_LIMITED_REPLY) == 1


def test_choice_button_round_trip_sends_the_browse_pick() -> None:
    choices = Choices(
        intro="Menus:",
        outro="Which?",
        cards=[ChoiceCard(name="ramen", image="c.png", group="the main event", category="ramen")],
    )
    harness = Harness(turn_reply=reply(choices=choices))
    harness.send("show me the menu")
    [keyboard_message] = [p for p in harness.client.sent() if "reply_markup" in p]
    button = keyboard_message["reply_markup"]["inline_keyboard"][0][0]
    assert button["text"] == "ramen"
    assert len(button["callback_data"].encode()) <= 64

    harness.tap(button["callback_data"])
    assert harness.client.sent("answerCallbackQuery") == [{"callback_query_id": "cb1"}]
    assert harness.turns[-1] == (
        "telegram:42",
        "Show me ramen from the main event",
        BrowsePick(group="the main event", category="ramen"),
    )


def test_unknown_button_says_it_expired() -> None:
    harness = Harness()
    harness.tap("no-such-key")
    assert harness.client.texts() == [EXPIRED_BUTTON_REPLY]
    assert harness.turns == []


def test_conversation_limit_points_at_reset_not_the_pages_button() -> None:
    harness = Harness(turn_reply=reply(CONVERSATION_LIMIT_REPLY))
    harness.send("hi")
    assert harness.client.texts() == [CONVERSATION_LIMIT_TELEGRAM_REPLY]


def test_photos_telegram_cannot_send_fall_back_to_their_captions() -> None:
    items = [dish("Vegan Ramen"), dish("Chicken Ramen")]
    harness = Harness(turn_reply=reply("Two.", cited=items), failing={"sendMediaGroup"})
    harness.send("ramen?")
    assert harness.client.texts()[-1] == "Vegan Ramen — £9.50\n\nChicken Ramen — £9.50"


def test_reply_busy_answers_the_chat() -> None:
    harness = Harness()
    harness.bot.reply_busy(
        {"update_id": 1, "message": {"chat": {"id": CHAT_ID, "type": "private"}, "text": "hi"}}
    )
    assert harness.client.texts() == [BUSY_REPLY]


def _png(color: tuple[int, int, int] = (200, 40, 40)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (240, 240), color).save(out, format="PNG")
    return out.getvalue()


GRID_CHOICES = Choices(
    intro="Our menu is organised into these categories:",
    outro="Which would you like to see?",
    cards=[
        ChoiceCard(
            name="drinks", image="https://img.example/menu/drinks/cover.png", group="drinks"
        ),
        ChoiceCard(name="sides", image="https://img.example/menu/sides/cover.png", group="sides"),
    ],
)


def test_layout_choices_with_picture_urls_become_a_grid() -> None:
    [grid] = layout(reply(choices=GRID_CHOICES))
    assert isinstance(grid, Grid)
    assert grid.cards == [(card.name, card.image) for card in GRID_CHOICES.cards]
    assert grid.text == f"{GRID_CHOICES.intro}\n\n{GRID_CHOICES.outro}"
    assert [b.label for b in grid.buttons] == ["drinks", "sides"]


def test_grid_is_uploaded_once_then_resent_by_file_id() -> None:
    covers = {card.image: _png() for card in GRID_CHOICES.cards}
    harness = Harness(turn_reply=reply(choices=GRID_CHOICES), covers=covers)
    harness.send("give me the menu")

    [(method, fields, files)] = harness.client.uploads
    assert method == "sendPhoto"
    assert fields["caption"] == f"{GRID_CHOICES.intro}\n\n{GRID_CHOICES.outro}"
    keyboard = fields["reply_markup"]["inline_keyboard"]
    assert [row[0]["text"] for row in keyboard] == ["drinks", "sides"]
    name, picture, mime = files["photo"]
    assert mime == "image/jpeg"
    with Image.open(io.BytesIO(picture)) as grid:
        assert grid.format == "JPEG"
    assert harness.client.texts() == []  # the text went as the caption

    harness.send("give me the menu")
    assert len(harness.client.uploads) == 1  # not drawn or uploaded again
    assert len(harness.client.fetched) == 2
    [resent] = harness.client.sent("sendPhoto")
    assert resent["photo"] == "grid-file-id"
    assert resent["caption"] == fields["caption"]


def test_grid_falls_back_to_text_and_buttons_when_no_cover_loads() -> None:
    harness = Harness(turn_reply=reply(choices=GRID_CHOICES))
    harness.send("give me the menu")
    assert harness.client.uploads == []
    [message] = harness.client.sent()
    assert message["text"] == f"{GRID_CHOICES.intro}\n\n{GRID_CHOICES.outro}"
    assert len(message["reply_markup"]["inline_keyboard"]) == 2


def test_grid_falls_back_to_text_and_buttons_when_the_upload_fails() -> None:
    covers = {card.image: _png() for card in GRID_CHOICES.cards}
    harness = Harness(turn_reply=reply(choices=GRID_CHOICES), covers=covers, failing={"upload"})
    harness.send("give me the menu")
    [message] = harness.client.sent()
    assert "reply_markup" in message
    assert not any(key.startswith("telegram:grid:") for key in harness.redis.store)


def test_build_grid_lays_out_tiles_with_room_for_names() -> None:
    names = ["desserts + sweet treats", "drinks", "a very long category name that must wrap"]
    picture = build_grid([(names[0], _png()), (names[1], None), (names[2], b"not an image")])
    with Image.open(io.BytesIO(picture)) as grid:
        # 3 cards: 3 columns of 240px tiles with 16px gaps, one row plus its label strip.
        assert grid.size == (3 * 240 + 4 * 16, 240 + 58 + 2 * 16)
        pixel = grid.convert("RGB").getpixel((16 + 120, 16 + 120))
        # the red cover is in the first tile
        assert isinstance(pixel, tuple) and pixel[0] > 150


# --- the Bot API client ---------------------------------------------------------------------


def test_client_returns_result_and_never_logs_the_token(caplog: pytest.LogCaptureFixture) -> None:
    token = "123:SECRET-TOKEN"
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/sendMessage"):
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 1}})
        return httpx.Response(400, json={"ok": False, "description": "Bad Request: nope"})

    client = TelegramClient(token, transport=httpx.MockTransport(handler))
    with caplog.at_level(logging.DEBUG):
        assert client.call("sendMessage", {"chat_id": 1, "text": "hi"}) == {"message_id": 1}
        assert client.call("sendPhoto", {"chat_id": 1}) is None
    assert seen[0].url.path == f"/bot{token}/sendMessage"
    assert json.loads(seen[0].content) == {"chat_id": 1, "text": "hi"}
    assert "Bad Request: nope" in caplog.text
    assert token not in caplog.text
    client.close()


def test_client_upload_sends_a_multipart_form_and_fetch_gets_pictures() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "img.example":
            if request.url.path == "/ok.png":
                return httpx.Response(200, content=b"PNG")
            return httpx.Response(404)
        return httpx.Response(200, json={"ok": True, "result": {"photo": []}})

    client = TelegramClient("t", transport=httpx.MockTransport(handler))
    result = client.upload(
        "sendPhoto",
        {"chat_id": 1, "caption": "hi", "reply_markup": {"inline_keyboard": []}},
        {"photo": ("menu.jpg", b"JPEGDATA", "image/jpeg")},
    )
    assert result == {"photo": []}
    body = seen[0].content
    assert seen[0].headers["content-type"].startswith("multipart/form-data")
    assert b'{"inline_keyboard": []}' in body and b"JPEGDATA" in body
    assert client.fetch("https://img.example/ok.png") == b"PNG"
    assert client.fetch("https://img.example/missing.png") is None
    client.close()


def test_client_swallows_network_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    client = TelegramClient("t", transport=httpx.MockTransport(handler))
    assert client.call("sendMessage", {}) is None
    client.close()


# --- the webhook route ----------------------------------------------------------------------


class FakeStateSnapshot:
    values: dict[str, Any] = {"history": []}


class FakeGraph:
    def get_state(self, config: dict[str, Any]) -> FakeStateSnapshot:
        return FakeStateSnapshot()

    def invoke(self, input: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {"answer": f"You asked: {input['question']}", "usage": {"total_tokens": 10}}


class FakeCheckpointer:
    def delete_thread(self, thread_id: str) -> None:
        pass


class FakeCostRedis(FakeRedis):
    """FakeRedis plus the pipeline app.cost_control uses to record token usage."""

    def pipeline(self) -> FakeCostRedis:
        return self

    def incrby(self, key: str, amount: int) -> FakeCostRedis:
        return self

    def execute(self) -> None:
        pass


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "weaviate_url": "",
        "weaviate_read_api_key": "",
        "image_base_url": "",
        "telegram_bot_token": "123:token",
        "telegram_webhook_secret": SECRET,
    }
    return Settings(**{**values, **overrides})


def _update(text: str = "hi") -> dict[str, Any]:
    return {"update_id": 1, "message": {"chat": {"id": CHAT_ID, "type": "private"}, "text": text}}


def test_webhook_is_404_while_the_bot_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "settings", _settings(telegram_bot_token=""))
    with TestClient(app) as client:
        response = client.post("/telegram/webhook", json=_update())
    assert response.status_code == 404


@pytest.mark.parametrize("headers", [{}, {"X-Telegram-Bot-Api-Secret-Token": "wrong"}])
def test_webhook_rejects_requests_without_the_secret(
    monkeypatch: pytest.MonkeyPatch, headers: dict[str, str]
) -> None:
    monkeypatch.setattr(main_module, "settings", _settings())
    with TestClient(app) as client:
        response = client.post("/telegram/webhook", json=_update(), headers=headers)
    assert response.status_code == 403


def test_webhook_rejects_everything_when_no_secret_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "settings", _settings(telegram_webhook_secret=""))
    with TestClient(app) as client:
        response = client.post(
            "/telegram/webhook", json=_update(), headers={"X-Telegram-Bot-Api-Secret-Token": ""}
        )
    assert response.status_code == 403


def test_webhook_answers_through_the_shared_chat_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "settings", _settings())
    telegram = FakeTelegramClient()
    with TestClient(app) as client:
        app.state.telegram_client = telegram
        app.state.graph = FakeGraph()
        app.state.redis_client = FakeCostRedis()
        app.state.checkpointer = FakeCheckpointer()
        response = client.post(
            "/telegram/webhook",
            json=_update("any ramen?"),
            headers={"X-Telegram-Bot-Api-Secret-Token": SECRET},
        )
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert telegram.texts() == ["You asked: any ramen?"]
