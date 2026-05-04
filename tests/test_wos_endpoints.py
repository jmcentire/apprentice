"""Integration tests for WOS HTTP endpoints in serve.py."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from apprentice.cli.serve import ApprenticeServer, SecurityConfig
from apprentice.package_runtime import PackageRuntimeState
from apprentice.skill_package import SkillPackage


def _build_mock_apprentice():
    """Build a mock Apprentice with all required components."""
    a = MagicMock()
    a._running = True
    a._start_time_utc = None
    a._skill_package = None
    a._package_runtime_state = None
    a._package_registry_store = None
    a._tool_preflight_client = None

    # Training data store
    a._training_data_store = MagicMock()
    a._training_data_store.add_example = MagicMock()

    # Audit log
    a._audit_log = MagicMock()
    a._audit_log.log = AsyncMock()

    # Confidence engine
    a._confidence_engine = MagicMock()
    a._confidence_engine.record_comparison = MagicMock()
    snap = MagicMock()
    snap.correlation_score = 0.85
    snap.phase = "COACHING"
    a._confidence_engine.get_snapshot = MagicMock(return_value=snap)

    # Task registry
    a._task_registry = MagicMock()
    a._task_registry.task_names = {"guest_response", "refund_handling"}

    # Run method
    response = MagicMock()
    response.output = {"text": "recommended response"}
    response.task_name = "guest_response"
    response.source = MagicMock(value="remote")
    response.status = MagicMock(value="success")
    response.request_id = "test-req-1"
    response.duration_ms = 42.0
    response.metadata = MagicMock(
        model_id="test-model",
        cost_usd=0.001,
        fallback_used=False,
    )
    a.run = AsyncMock(return_value=response)

    return a


async def _send_request(server, method, path, body=None, headers=None):
    """Simulate an HTTP request to the server."""
    # Build raw HTTP request
    header_lines = []
    if headers:
        for k, v in headers.items():
            header_lines.append(f"{k}: {v}")

    body_str = ""
    if body is not None:
        body_str = json.dumps(body) if isinstance(body, dict) else body
        header_lines.append(f"Content-Length: {len(body_str)}")
        header_lines.append("Content-Type: application/json")

    raw = f"{method} {path} HTTP/1.1\r\n"
    raw += "\r\n".join(header_lines)
    raw += f"\r\n\r\n{body_str}"

    # Mock reader/writer
    reader = AsyncMock()
    reader.read = AsyncMock(return_value=raw.encode("utf-8"))

    writer = MagicMock()
    writer.get_extra_info = MagicMock(return_value=("127.0.0.1", 12345))
    writer.write = MagicMock()
    writer.drain = AsyncMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()

    await server._handle_connection(reader, writer)

    # Parse response
    calls = writer.write.call_args_list
    if not calls:
        return None, None

    response_bytes = calls[0][0][0]
    response_text = response_bytes.decode("utf-8")

    # Split headers and body
    parts = response_text.split("\r\n\r\n", 1)
    status_line = parts[0].split("\r\n")[0]
    status_code = int(status_line.split(" ")[1])
    body_text = parts[1] if len(parts) > 1 else ""

    return status_code, json.loads(body_text) if body_text else None


# ---------------------------------------------------------------------------
# Event endpoint tests
# ---------------------------------------------------------------------------


class TestEventsEndpoint:
    """Tests for POST /v1/events."""

    @pytest.fixture
    def server(self):
        apprentice = _build_mock_apprentice()
        return ApprenticeServer(apprentice, pipeline_interval=9999)

    async def test_valid_event_returns_200(self, server):
        status, body = await _send_request(server, "POST", "/v1/events", {
            "event_type": "message_sent",
            "conversation_id": "conv-123",
            "agent_id": "agent-1",
            "organization_id": "org-1",
            "timestamp": "2024-01-01T00:00:00Z",
            "payload": {"message_id": "msg-1"},
        })
        assert status == 200
        assert body["status"] == "accepted"

    async def test_malformed_json_returns_400(self, server):
        status, body = await _send_request(server, "POST", "/v1/events", "not json")
        assert status == 400

    async def test_fire_and_forget_on_internal_error(self):
        """Events should return 200 even when handler errors internally."""
        broken = _build_mock_apprentice()
        broken._training_data_store = None
        broken._audit_log = MagicMock()
        broken._audit_log.log = AsyncMock(side_effect=RuntimeError("boom"))
        server = ApprenticeServer(broken, pipeline_interval=9999)

        status, body = await _send_request(server, "POST", "/v1/events", {
            "event_type": "task_created",
            "conversation_id": "conv-1",
            "timestamp": "2024-01-01T00:00:00Z",
        })
        assert status == 200
        assert body["status"] == "accepted"

    async def test_minimal_event_accepted(self, server):
        """Event with only required fields should be accepted."""
        status, body = await _send_request(server, "POST", "/v1/events", {
            "event_type": "agent_assigned",
            "conversation_id": "c-1",
            "timestamp": "2024-01-01T00:00:00Z",
        })
        assert status == 200


# ---------------------------------------------------------------------------
# Feedback endpoint tests
# ---------------------------------------------------------------------------


class TestFeedbackEndpoint:
    """Tests for POST /v1/feedback."""

    @pytest.fixture
    def server(self):
        apprentice = _build_mock_apprentice()
        return ApprenticeServer(apprentice, pipeline_interval=9999)

    async def test_valid_feedback_returns_200(self, server):
        status, body = await _send_request(server, "POST", "/v1/feedback", {
            "request_id": "req-1",
            "skill": "guest_response",
            "feedback_type": "accept",
        })
        assert status == 200
        assert body["status"] == "recorded"
        assert body["match_score"] == 1.0

    async def test_reject_feedback(self, server):
        status, body = await _send_request(server, "POST", "/v1/feedback", {
            "request_id": "req-2",
            "skill": "refund_handling",
            "feedback_type": "reject",
        })
        assert status == 200
        assert body["match_score"] == 0.0

    async def test_edit_feedback(self, server):
        status, body = await _send_request(server, "POST", "/v1/feedback", {
            "request_id": "req-3",
            "skill": "ticket_triage",
            "feedback_type": "edit",
            "edited_output": {"text": "revised"},
        })
        assert status == 200
        assert body["match_score"] == 0.5

    async def test_malformed_json_returns_400(self, server):
        status, body = await _send_request(server, "POST", "/v1/feedback", "{bad")
        assert status == 400


# ---------------------------------------------------------------------------
# Recommendations endpoint tests
# ---------------------------------------------------------------------------


class TestRecommendationsEndpoint:
    """Tests for POST /v1/recommendations."""

    @pytest.fixture
    def server(self):
        apprentice = _build_mock_apprentice()
        return ApprenticeServer(apprentice, pipeline_interval=9999)

    async def test_valid_recommend_returns_200(self, server):
        status, body = await _send_request(server, "POST", "/v1/recommendations", {
            "skill": "guest_response",
            "context": {"conversation_id": "conv-1", "message": "hello"},
            "request_id": "req-1",
        })
        assert status == 200
        assert body["skill_name"] == "guest_response"
        assert body["recommendation_id"] == "req-1"
        assert body["confidence"] == 0.85

    async def test_recommend_without_request_id(self, server):
        status, body = await _send_request(server, "POST", "/v1/recommendations", {
            "skill": "refund_handling",
            "context": {},
        })
        assert status == 200
        assert body["recommendation_id"]  # auto-generated

    async def test_recommend_run_error_returns_503(self):
        apprentice = _build_mock_apprentice()
        apprentice.run = AsyncMock(side_effect=RuntimeError("model unavailable"))
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "POST", "/v1/recommendations", {
            "skill": "guest_response",
            "context": {},
        })
        assert status == 503

    async def test_malformed_json_returns_400(self, server):
        status, body = await _send_request(server, "POST", "/v1/recommendations", "{{bad")
        assert status == 400


# ---------------------------------------------------------------------------
# Skills endpoint tests
# ---------------------------------------------------------------------------


class TestSkillsEndpoint:
    """Tests for GET /v1/skills."""

    @pytest.fixture
    def server(self):
        apprentice = _build_mock_apprentice()
        return ApprenticeServer(apprentice, pipeline_interval=9999)

    async def test_list_skills_returns_200(self, server):
        status, body = await _send_request(server, "GET", "/v1/skills")
        assert status == 200
        assert "skills" in body
        assert "timestamp" in body
        assert len(body["skills"]) == 2
        names = {s["name"] for s in body["skills"]}
        assert names == {"guest_response", "refund_handling"}

    async def test_list_skills_uses_registered_skill_package(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            skills=[
                {
                    "name": "classify_ticket",
                    "description": "Classify tickets.",
                    "methods": [{"name": "run", "kind": "http", "target": "/v1/run"}],
                    "actions": [{"name": "propose_classification"}],
                    "outcomes": [{"name": "accepted"}],
                }
            ],
        )
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "GET", "/v1/skills")

        assert status == 200
        assert body["skills"][0]["name"] == "classify_ticket"
        assert body["skills"][0]["methods"][0]["target"] == "/v1/run"


class TestSkillPackageEndpoint:
    async def test_skill_package_endpoint_returns_registered_package(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            skills=[{"name": "classify_ticket"}],
        )
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "GET", "/v1/skill-package")

        assert status == 200
        assert body["package_id"] == "example.support"

    async def test_packaged_event_maps_to_training_example(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            skills=[{"name": "classify_ticket"}],
            event_mappings=[
                {
                    "event_type": "ticket.classified",
                    "skill": "classify_ticket",
                    "input_path": "payload.input",
                    "output_path": "payload.output",
                    "subject_path": "payload.ticket_id",
                }
            ],
        )
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "POST", "/v1/events", {
            "event_type": "ticket.classified",
            "timestamp": "2026-05-03T22:00:00Z",
            "payload": {
                "ticket_id": "ticket-1",
                "input": {"text": "card declined"},
                "output": {"category": "billing"},
            },
        })

        assert status == 200
        assert body["mapped"] is True
        assert body["skill"] == "classify_ticket"
        apprentice._training_data_store.add_example.assert_called_once()

    async def test_constraint_check_uses_registered_package(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            skills=[
                {
                    "name": "classify_ticket",
                    "constraints": [
                        {
                            "name": "has_text",
                            "kind": "hard",
                            "expression": "not_empty(input.text)",
                        }
                    ],
                    "actions": [
                        {
                            "name": "propose_classification",
                            "constraints": ["has_text"],
                        }
                    ],
                }
            ],
        )
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "POST", "/v1/constraints/check", {
            "skill": "tenant-1:classify_ticket",
            "action": "propose_classification",
            "input": {"text": ""},
        })

        assert status == 200
        assert body["allowed"] is False
        assert body["violations"][0]["name"] == "has_text"

    async def test_run_blocks_on_hard_package_constraint(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            skills=[
                {
                    "name": "classify_ticket",
                    "methods": [{"name": "run", "kind": "http", "target": "/v1/run"}],
                    "constraints": [
                        {
                            "name": "has_text",
                            "kind": "hard",
                            "expression": "not_empty(input.text)",
                        }
                    ],
                    "actions": [
                        {
                            "name": "propose_classification",
                            "constraints": ["has_text"],
                        }
                    ],
                }
            ],
        )
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "POST", "/v1/run", {
            "skill": "tenant-1:classify_ticket",
            "method": "run",
            "action": "propose_classification",
            "input": {"text": ""},
        })

        assert status == 422
        assert body["status"] == "blocked"
        assert body["violations"][0]["name"] == "has_text"
        apprentice.run.assert_not_called()

    async def test_labeled_example_accepts_tenant_qualified_skill(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            skills=[{"name": "classify_ticket"}],
        )
        apprentice._package_runtime_state = PackageRuntimeState()
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "POST", "/v1/labeled-examples", {
            "skill": "tenant-1:classify_ticket",
            "request_id": "action-1",
            "input": {"text": "card declined"},
            "draft_output": {"category": "technical"},
            "final_output": {"category": "billing"},
            "feedback_type": "edit",
        })

        assert status == 200
        assert body["status"] == "recorded"
        assert body["match_score"] == 0.5
        apprentice._training_data_store.add_example.assert_called_once()

        status, status_body = await _send_request(server, "GET", "/v1/status")
        package_tasks = [task for task in status_body["tasks"] if task.get("task_name") == "tenant-1:classify_ticket"]
        assert status == 200
        assert package_tasks[0]["confidence"] == 0.5

    async def test_package_registry_endpoint_returns_compact_metadata(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            skills=[
                {
                    "name": "classify_ticket",
                    "methods": [{"name": "run", "kind": "http", "target": "/v1/run"}],
                    "tools": [{"name": "ticket_lookup", "kind": "http", "target": "https://example.local"}],
                    "evaluators": [{"name": "json_match", "kind": "json_schema"}],
                    "artifacts": [{"name": "ticket_text", "modality": "text", "path": "input.text"}],
                }
            ],
        )
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "GET", "/v1/package-registry")

        assert status == 200
        assert body["package_id"] == "example.support"
        assert body["schema_version"] == 1
        assert body["skills"][0]["tools"] == ["ticket_lookup"]
        assert body["skills"][0]["evaluators"] == ["json_match"]

    async def test_tools_endpoint_returns_runtime_resolved_tools(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            runtime={
                "tool_endpoint_overrides": {
                    "ticket_lookup": "https://prod.example/tickets/{ticket_id}",
                }
            },
            skills=[
                {
                    "name": "classify_ticket",
                    "tools": [
                        {
                            "name": "ticket_lookup",
                            "kind": "http",
                            "target": "https://dev.example/tickets/{ticket_id}",
                        }
                    ],
                }
            ],
        )
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "GET", "/v1/tools")

        assert status == 200
        assert body["tools"][0]["skill"] == "classify_ticket"
        assert body["tools"][0]["target"] == "https://prod.example/tickets/{ticket_id}"

    async def test_tool_call_is_blocked_without_preflight(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            skills=[
                {
                    "name": "classify_ticket",
                    "tools": [
                        {
                            "name": "custom_lookup",
                            "kind": "mcp",
                            "target": "tools/custom_lookup",
                        }
                    ],
                }
            ],
        )
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "POST", "/v1/tools/call", {
            "skill": "classify_ticket",
            "tool": "custom_lookup",
            "input": {"ticket_id": "ticket-1"},
        })

        assert status == 200
        assert body["status"] == "blocked"

    async def test_evaluator_run_scores_builtin_match(self):
        apprentice = _build_mock_apprentice()
        apprentice._skill_package = SkillPackage(
            package_id="example.support",
            skills=[
                {
                    "name": "classify_ticket",
                    "evaluators": [{"name": "json_match", "kind": "json_schema"}],
                }
            ],
        )
        server = ApprenticeServer(apprentice, pipeline_interval=9999)

        status, body = await _send_request(server, "POST", "/v1/evaluators/run", {
            "skill": "classify_ticket",
            "evaluator": "json_match",
            "candidate": {"category": "billing"},
            "reference": {"category": "billing"},
        })

        assert status == 200
        assert body["status"] == "ok"
        assert body["score"] == 1.0


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


class TestAuth:
    """Test authentication on WOS endpoints."""

    async def test_events_with_api_key_auth(self):
        apprentice = _build_mock_apprentice()
        security = SecurityConfig(auth_mode="api-key", api_key="test-secret")
        server = ApprenticeServer(apprentice, pipeline_interval=9999, security=security)

        # Without auth: 401
        status, _ = await _send_request(server, "POST", "/v1/events", {
            "event_type": "message_sent",
            "conversation_id": "c-1",
            "timestamp": "2024-01-01T00:00:00Z",
        })
        assert status == 401

        # With auth: 200
        status, body = await _send_request(server, "POST", "/v1/events", {
            "event_type": "message_sent",
            "conversation_id": "c-1",
            "timestamp": "2024-01-01T00:00:00Z",
        }, headers={"Authorization": "Bearer test-secret"})
        assert status == 200

    async def test_health_bypasses_auth(self):
        apprentice = _build_mock_apprentice()
        security = SecurityConfig(auth_mode="api-key", api_key="test-secret")
        server = ApprenticeServer(apprentice, pipeline_interval=9999, security=security)

        status, _ = await _send_request(server, "GET", "/health")
        assert status == 200
