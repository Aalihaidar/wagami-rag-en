from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


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
    assert "/static/css/chat.css" in body


def test_static_assets_are_served() -> None:
    js_response = client.get("/static/js/chat.js")
    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]

    css_response = client.get("/static/css/chat.css")
    assert css_response.status_code == 200
    assert "css" in css_response.headers["content-type"]
