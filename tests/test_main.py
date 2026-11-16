import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import Settings
from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _mini_app_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """The header tests below mean the default (no Mini App URL) regardless of this machine's own
    .env; the Mini App test sets the URL itself."""
    monkeypatch.setattr(main_module, "settings", Settings(telegram_web_app_url=""))


def test_healthz() -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_security_headers_present_on_every_response() -> None:
    response = client.get("/healthz")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    # HSTS is production-only (see app/main.py's security_headers) -- the test app runs with
    # the default development APP_ENV.
    assert "Strict-Transport-Security" not in response.headers


def test_no_framing_and_no_external_scripts_while_the_mini_app_is_off() -> None:
    response = client.get("/healthz")
    csp = response.headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in csp
    assert "telegram.org" not in csp


def test_telegram_may_frame_the_page_once_a_mini_app_url_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main_module, "settings", Settings(telegram_web_app_url="https://chat.example.com")
    )
    response = client.get("/healthz")
    csp = response.headers["Content-Security-Policy"]
    # Only Telegram's own domains may frame it, and X-Frame-Options (which cannot name them) is
    # gone; everything else in the policy is unchanged.
    assert "frame-ancestors https://web.telegram.org https://*.telegram.org;" in csp
    assert "frame-ancestors 'none'" not in csp
    assert "script-src 'self' https://telegram.org" in csp
    assert "default-src 'self'" in csp and "style-src 'self'" in csp
    assert "X-Frame-Options" not in response.headers


def _css_declarations(css: str, selector: str) -> dict[str, str]:
    """The `--token: value` pairs of the first rule with exactly this selector."""
    start = css.index(selector + " {") + len(selector) + 2
    body = css[start : css.index("}", start)]
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", body))


def test_forced_dark_palette_is_identical_to_the_device_dark_palette() -> None:
    """Telegram's theme is applied by data-theme="dark", a second copy of the dark tokens that
    the device setting (prefers-color-scheme) uses; the two must never drift apart."""
    css = (Path(main_module.__file__).parent / "static" / "css" / "chat.css").read_text()
    device = _css_declarations(css, ':root:not([data-theme="light"])')
    forced = _css_declarations(css, ':root[data-theme="dark"]')
    assert len(device) >= 15
    assert forced == device


def test_chat_css_defines_every_dark_token_in_the_light_palette_too() -> None:
    css = (Path(main_module.__file__).parent / "static" / "css" / "chat.css").read_text()
    light = _css_declarations(css, ":root")
    assert set(_css_declarations(css, ':root[data-theme="dark"]')) <= set(light)


def test_unknown_route_returns_consistent_error_shape() -> None:
    response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404
    assert response.json() == {"error": "Not Found"}


def test_chat_page_renders() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    body = response.text
    assert "<title>Menu &amp; FAQ Assistant</title>" in body
    assert "/static/js/chat.js" in body
    assert "/static/js/telegram.js" in body
    assert "/static/css/chat.css" in body


def test_static_assets_are_served() -> None:
    js_response = client.get("/static/js/chat.js")
    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]

    css_response = client.get("/static/css/chat.css")
    assert css_response.status_code == 200
    assert "css" in css_response.headers["content-type"]
