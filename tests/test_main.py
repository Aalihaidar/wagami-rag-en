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
