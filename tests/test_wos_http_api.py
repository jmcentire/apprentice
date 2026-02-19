"""Tests for apprentice.wos_http_api — WOS Copilot HTTP API."""

import time
import pytest

from starlette.testclient import TestClient

from apprentice.wos_http_api import (
    WOSCopilotConfig,
    WOSCopilotService,
    APIKeyAuth,
    RateLimiter,
    create_app,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

API_KEY = "test-secret-key"


@pytest.fixture
def config(tmp_path):
    return WOSCopilotConfig(
        api_key=API_KEY,
        rate_limit_per_minute=60,
        observer_enabled=True,
        observer_context_window=50,
        observer_shadow_rate=0.0,
        feedback_enabled=True,
        feedback_storage_dir=str(tmp_path / "feedback"),
    )


@pytest.fixture
def app(config):
    return create_app(config)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-Apprentice-Key": API_KEY}


@pytest.fixture
def triage_context():
    return {"conversation_id": "c1", "conversation_text": "My AC is broken"}


@pytest.fixture
def refund_context():
    return {
        "conversation_id": "c1",
        "conversation_text": "I want a refund",
        "booking_id": "b1",
        "booking_value": 500.0,
    }


@pytest.fixture
def wos_event_body():
    return {
        "event_type": "message_sent",
        "conversation_id": "conv-1",
        "agent_id": "agent-1",
        "organization_id": "org-1",
        "timestamp": "2024-01-01T00:00:00Z",
        "payload": {"text": "Hello!"},
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:
    def test_create_valid(self, tmp_path):
        c = WOSCopilotConfig(api_key="key", feedback_storage_dir=str(tmp_path))
        assert c.api_key == "key"
        assert c.rate_limit_per_minute == 60
        assert c.version == "0.1.0"

    def test_frozen(self, tmp_path):
        c = WOSCopilotConfig(api_key="key", feedback_storage_dir=str(tmp_path))
        with pytest.raises(Exception):
            c.api_key = "other"

    def test_defaults(self):
        c = WOSCopilotConfig(api_key="key")
        assert c.observer_enabled is True
        assert c.observer_context_window == 50
        assert c.observer_shadow_rate == 0.1
        assert c.feedback_enabled is True


# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_under_limit(self):
        rl = RateLimiter(max_requests_per_minute=5)
        for _ in range(5):
            assert rl.is_allowed("org-1") is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests_per_minute=3)
        for _ in range(3):
            rl.is_allowed("org-1")
        assert rl.is_allowed("org-1") is False

    def test_separate_orgs(self):
        rl = RateLimiter(max_requests_per_minute=2)
        rl.is_allowed("org-1")
        rl.is_allowed("org-1")
        assert rl.is_allowed("org-1") is False
        assert rl.is_allowed("org-2") is True

    def test_default_limit(self):
        rl = RateLimiter()
        for _ in range(60):
            assert rl.is_allowed("org-1") is True
        assert rl.is_allowed("org-1") is False


# ---------------------------------------------------------------------------
# API Key Auth
# ---------------------------------------------------------------------------

class TestAPIKeyAuth:
    def test_valid_key(self, client):
        from starlette.testclient import TestClient
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        auth = APIKeyAuth("secret")

        async def test_route(request: Request) -> JSONResponse:
            if auth.is_valid(request):
                return JSONResponse({"ok": True})
            return JSONResponse({"ok": False}, status_code=401)

        test_app = Starlette(routes=[Route("/test", test_route)])
        tc = TestClient(test_app)

        resp = tc.get("/test", headers={"X-Apprentice-Key": "secret"})
        assert resp.status_code == 200

        resp = tc.get("/test", headers={"X-Apprentice-Key": "wrong"})
        assert resp.status_code == 401

        resp = tc.get("/test")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Health endpoint (no auth)
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_no_auth(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["skills_loaded"] == 5
        assert "version" in data

    def test_health_with_auth(self, client, auth_headers):
        resp = client.get("/api/v1/health", headers=auth_headers)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Events endpoint
# ---------------------------------------------------------------------------

class TestEventsEndpoint:
    def test_ingest_event(self, client, auth_headers, wos_event_body):
        resp = client.post("/api/v1/events", json=wos_event_body, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_ingest_invalid_event(self, client, auth_headers):
        resp = client.post("/api/v1/events", json={"bad": "data"}, headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["status"] == "accepted"

    def test_ingest_no_auth(self, client, wos_event_body):
        resp = client.post("/api/v1/events", json=wos_event_body)
        assert resp.status_code == 401

    def test_ingest_bad_json(self, client, auth_headers):
        resp = client.post(
            "/api/v1/events",
            content=b"not json",
            headers={**auth_headers, "content-type": "application/json"},
        )
        assert resp.status_code == 200

    def test_fire_and_forget_semantics(self, client, auth_headers):
        for body in [
            {},
            {"event_type": "unknown"},
            {"event_type": "message_sent"},
        ]:
            resp = client.post("/api/v1/events", json=body, headers=auth_headers)
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Recommend endpoint
# ---------------------------------------------------------------------------

class TestRecommendEndpoint:
    def test_triage_recommend(self, client, auth_headers, triage_context):
        resp = client.post(
            "/api/v1/recommend",
            json={"skill": "ticket_triage", "context": triage_context},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["skill_name"] == "ticket_triage"
        assert "category" in data
        assert "confidence" in data

    def test_refund_recommend(self, client, auth_headers, refund_context):
        resp = client.post(
            "/api/v1/recommend",
            json={"skill": "refund_handling", "context": refund_context},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["skill_name"] == "refund_handling"

    def test_unknown_skill(self, client, auth_headers):
        resp = client.post(
            "/api/v1/recommend",
            json={"skill": "nonexistent", "context": {"conversation_id": "c1"}},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    def test_missing_context(self, client, auth_headers):
        resp = client.post(
            "/api/v1/recommend",
            json={"skill": "ticket_triage", "context": {"conversation_id": "c1"}},
            headers=auth_headers,
        )
        assert resp.status_code == 422

    def test_missing_skill_field(self, client, auth_headers):
        resp = client.post(
            "/api/v1/recommend",
            json={"context": {}},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_missing_context_field(self, client, auth_headers):
        resp = client.post(
            "/api/v1/recommend",
            json={"skill": "ticket_triage"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_no_auth(self, client, triage_context):
        resp = client.post(
            "/api/v1/recommend",
            json={"skill": "ticket_triage", "context": triage_context},
        )
        assert resp.status_code == 401

    def test_custom_request_id(self, client, auth_headers, triage_context):
        resp = client.post(
            "/api/v1/recommend",
            json={
                "skill": "ticket_triage",
                "context": triage_context,
                "request_id": "custom-123",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_all_five_skills(self, client, auth_headers):
        contexts = {
            "ticket_triage": {"conversation_id": "c", "conversation_text": "t"},
            "guest_response": {"conversation_id": "c", "conversation_text": "t", "booking_id": "b"},
            "refund_handling": {"conversation_id": "c", "conversation_text": "t", "booking_id": "b", "booking_value": 100},
            "booking_modification": {"conversation_id": "c", "conversation_text": "t", "booking_id": "b", "availability": {}},
            "escalation_detection": {"conversation_id": "c", "conversation_text": "t"},
        }
        for skill, ctx in contexts.items():
            resp = client.post(
                "/api/v1/recommend",
                json={"skill": skill, "context": ctx},
                headers=auth_headers,
            )
            assert resp.status_code == 200, f"Failed for skill {skill}"
            assert resp.json()["skill_name"] == skill


# ---------------------------------------------------------------------------
# Feedback endpoint
# ---------------------------------------------------------------------------

class TestFeedbackEndpoint:
    def test_submit_accept(self, client, auth_headers):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "request_id": "req-1",
                "skill": "ticket_triage",
                "feedback_type": "accept",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"

    def test_submit_reject_with_reason(self, client, auth_headers):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "request_id": "req-2",
                "skill": "ticket_triage",
                "feedback_type": "reject",
                "reason": "Wrong category",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_submit_edit_with_output(self, client, auth_headers):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "request_id": "req-3",
                "skill": "ticket_triage",
                "feedback_type": "edit",
                "edited_output": {"category": "complaint"},
            },
            headers=auth_headers,
        )
        assert resp.status_code == 200

    def test_invalid_feedback_type(self, client, auth_headers):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "request_id": "req-4",
                "skill": "ticket_triage",
                "feedback_type": "invalid",
            },
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_missing_fields(self, client, auth_headers):
        resp = client.post(
            "/api/v1/feedback",
            json={"request_id": "req-5"},
            headers=auth_headers,
        )
        assert resp.status_code == 400

    def test_no_auth(self, client):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "request_id": "req-6",
                "skill": "ticket_triage",
                "feedback_type": "accept",
            },
        )
        assert resp.status_code == 401

    def test_all_feedback_types(self, client, auth_headers):
        for ft in ["accept", "reject", "edit", "ignore", "ai_score"]:
            resp = client.post(
                "/api/v1/feedback",
                json={
                    "request_id": f"req-{ft}",
                    "skill": "ticket_triage",
                    "feedback_type": ft,
                },
                headers=auth_headers,
            )
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Skill status endpoint
# ---------------------------------------------------------------------------

class TestSkillStatusEndpoint:
    def test_get_status(self, client, auth_headers):
        resp = client.get("/api/v1/status/ticket_triage", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["skill"] == "ticket_triage"
        assert "confidence" in data
        assert "phase" in data
        assert "mode" in data
        assert "risk_level" in data

    def test_unknown_skill(self, client, auth_headers):
        resp = client.get("/api/v1/status/nonexistent", headers=auth_headers)
        assert resp.status_code == 404

    def test_no_auth(self, client):
        resp = client.get("/api/v1/status/ticket_triage")
        assert resp.status_code == 401

    def test_all_skills_status(self, client, auth_headers):
        skills = ["ticket_triage", "guest_response", "refund_handling",
                  "booking_modification", "escalation_detection"]
        for skill in skills:
            resp = client.get(f"/api/v1/status/{skill}", headers=auth_headers)
            assert resp.status_code == 200
            assert resp.json()["skill"] == skill


# ---------------------------------------------------------------------------
# Skills list endpoint
# ---------------------------------------------------------------------------

class TestSkillsEndpoint:
    def test_list_skills(self, client, auth_headers):
        resp = client.get("/api/v1/skills", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "skills" in data
        assert len(data["skills"]) == 5
        skill_names = {s["name"] for s in data["skills"]}
        assert "ticket_triage" in skill_names
        assert "guest_response" in skill_names

    def test_skills_structure(self, client, auth_headers):
        resp = client.get("/api/v1/skills", headers=auth_headers)
        for skill in resp.json()["skills"]:
            assert "name" in skill
            assert "risk_level" in skill
            assert "mode" in skill
            assert "confidence" in skill

    def test_no_auth(self, client):
        resp = client.get("/api/v1/skills")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Rate limiting integration
# ---------------------------------------------------------------------------

class TestRateLimitingIntegration:
    def test_rate_limited_recommend(self, tmp_path):
        config = WOSCopilotConfig(
            api_key=API_KEY,
            rate_limit_per_minute=3,
            feedback_storage_dir=str(tmp_path / "feedback"),
        )
        app = create_app(config)
        tc = TestClient(app)
        headers = {"X-Apprentice-Key": API_KEY}
        ctx = {"conversation_id": "c", "conversation_text": "t", "organization_id": "org-1"}

        for _ in range(3):
            resp = tc.post(
                "/api/v1/recommend",
                json={"skill": "ticket_triage", "context": ctx},
                headers=headers,
            )
            assert resp.status_code == 200

        resp = tc.post(
            "/api/v1/recommend",
            json={"skill": "ticket_triage", "context": ctx},
            headers=headers,
        )
        assert resp.status_code == 429

    def test_rate_limited_events(self, tmp_path):
        config = WOSCopilotConfig(
            api_key=API_KEY,
            rate_limit_per_minute=2,
            feedback_storage_dir=str(tmp_path / "feedback"),
        )
        app = create_app(config)
        tc = TestClient(app)
        headers = {"X-Apprentice-Key": API_KEY}
        event = {
            "event_type": "message_sent",
            "conversation_id": "conv-1",
            "organization_id": "org-1",
            "timestamp": "2024-01-01T00:00:00Z",
        }

        for _ in range(2):
            tc.post("/api/v1/events", json=event, headers=headers)

        resp = tc.post("/api/v1/events", json=event, headers=headers)
        assert resp.status_code == 429


# ---------------------------------------------------------------------------
# WOSCopilotService unit tests
# ---------------------------------------------------------------------------

class TestWOSCopilotService:
    def test_handle_event_valid(self, config):
        svc = WOSCopilotService(config)
        result = svc.handle_event({
            "event_type": "message_sent",
            "conversation_id": "conv-1",
            "organization_id": "org-1",
            "timestamp": "2024-01-01T00:00:00Z",
        })
        assert result["status"] == "accepted"

    def test_handle_event_invalid(self, config):
        svc = WOSCopilotService(config)
        result = svc.handle_event({"bad": "data"})
        assert result["status"] == "accepted"

    def test_handle_recommend(self, config):
        svc = WOSCopilotService(config)
        result = svc.handle_recommend(
            "ticket_triage",
            {"conversation_id": "c1", "conversation_text": "test"},
        )
        assert result["skill_name"] == "ticket_triage"

    def test_handle_recommend_unknown_skill(self, config):
        svc = WOSCopilotService(config)
        with pytest.raises(KeyError):
            svc.handle_recommend("nonexistent", {})

    def test_handle_feedback_accept(self, config):
        svc = WOSCopilotService(config)
        result = svc.handle_feedback({
            "request_id": "r1",
            "skill": "ticket_triage",
            "feedback_type": "accept",
        })
        assert result["status"] == "recorded"

    def test_handle_feedback_missing_fields(self, config):
        svc = WOSCopilotService(config)
        with pytest.raises(ValueError):
            svc.handle_feedback({})

    def test_handle_feedback_invalid_type(self, config):
        svc = WOSCopilotService(config)
        with pytest.raises(ValueError):
            svc.handle_feedback({
                "request_id": "r1",
                "skill": "ticket_triage",
                "feedback_type": "bad",
            })

    def test_get_skill_status(self, config):
        svc = WOSCopilotService(config)
        result = svc.get_skill_status("ticket_triage")
        assert result["skill"] == "ticket_triage"

    def test_get_skill_status_unknown(self, config):
        svc = WOSCopilotService(config)
        with pytest.raises(KeyError):
            svc.get_skill_status("nonexistent")

    def test_list_skills(self, config):
        svc = WOSCopilotService(config)
        result = svc.list_skills()
        assert len(result) == 5

    def test_health(self, config):
        svc = WOSCopilotService(config)
        result = svc.health()
        assert result["status"] == "healthy"
        assert result["skills_loaded"] == 5


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

class TestAppFactory:
    def test_create_app(self, config):
        app = create_app(config)
        assert app is not None
        assert hasattr(app.state, "service")
        assert hasattr(app.state, "auth")
        assert hasattr(app.state, "rate_limiter")

    def test_routes_registered(self, config):
        app = create_app(config)
        route_paths = {r.path for r in app.routes}
        assert "/api/v1/health" in route_paths
        assert "/api/v1/events" in route_paths
        assert "/api/v1/recommend" in route_paths
        assert "/api/v1/feedback" in route_paths
        assert "/api/v1/skills" in route_paths
        assert "/api/v1/status/{skill}" in route_paths
