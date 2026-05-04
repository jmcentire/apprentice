import os
from pathlib import Path

import pytest
import yaml

from apprentice.package_runtime import (
    AllowBuiltinToolPreflightClient,
    CredentialResolver,
    PackageRegistryStore,
    ToolPreflightClient,
    ToolPreflightResult,
    diff_packages,
    invoke_tool,
    package_fingerprint,
    run_evaluator,
    validate_package_runtime,
)
from apprentice.skill_package import EvaluatorSpec, SkillPackage, ToolSpec, load_skill_package


def make_package_dict() -> dict:
    return {
        "package_id": "example.support",
        "version": 1,
        "skills": [
            {
                "name": "classify_ticket",
                "outcomes": [{"name": "accepted"}],
                "constraints": [{"name": "has_text", "expression": "not_empty(input.text)"}],
                "actions": [{"name": "propose_classification", "constraints": ["has_text"]}],
                "tools": [
                    {
                        "name": "ticket_lookup",
                        "kind": "http",
                        "target": "https://support.example/tickets/{ticket_id}",
                    }
                ],
            }
        ],
        "runtime": {"tool_endpoint_overrides": {"ticket_lookup": "https://prod.example/tickets/{ticket_id}"}},
        "event_mappings": [{"event_type": "ticket.classified", "skill": "classify_ticket"}],
    }


def test_package_fingerprint_is_stable():
    first = SkillPackage(**make_package_dict())
    second = SkillPackage(**make_package_dict())

    assert package_fingerprint(first) == package_fingerprint(second)


def test_validate_package_runtime_reports_missing_env_credential():
    data = make_package_dict()
    data["runtime"]["auth_refs"] = {"ticket_lookup": "MISSING_TICKET_TOKEN"}
    package = SkillPackage(**data)

    report = validate_package_runtime(package, env={})

    assert report.ok is True
    assert report.diagnostics[0].level == "warning"
    assert "env:MISSING_TICKET_TOKEN" in report.diagnostics[0].message


def test_credential_resolver_redacts_values(monkeypatch):
    monkeypatch.setitem(os.environ, "TOKEN_FOR_TEST", "abcdef123456")

    resolved = CredentialResolver(os.environ).diagnose("TOKEN_FOR_TEST")

    assert resolved.configured is True
    assert resolved.redacted == "ab***56"


def test_diff_packages_marks_removed_skills_as_breaking():
    old = SkillPackage(**make_package_dict())
    data = make_package_dict()
    data["skills"][0]["name"] = "new_skill"
    data["event_mappings"] = []
    new = SkillPackage(**data)

    diff = diff_packages(old, new)

    assert diff["skills_removed"] == ["classify_ticket"]
    assert diff["breaking"] is True
    assert diff["breaking_changes"] == ["classify_ticket"]


def test_package_registry_store_persists_active_metadata(tmp_path: Path):
    package = SkillPackage(**make_package_dict())
    store = PackageRegistryStore(tmp_path / "package_registry.json")

    record = store.publish(package, environment="prod", overlays=["prod.yaml"])

    loaded = store.load()
    assert loaded == record
    assert loaded["package_id"] == "example.support"
    assert loaded["environment"] == "prod"
    assert loaded["overlays"] == ["prod.yaml"]


async def test_http_tool_is_blocked_without_preflight():
    tool = ToolSpec(name="lookup", kind="http", target="https://example.invalid/{id}")

    result = await invoke_tool(tool, {"id": "1"})

    assert result.status == "blocked"
    assert "no preflight client" in result.error


async def test_builtin_tool_is_allowed_by_builtin_preflight():
    tool = ToolSpec(name="echo", kind="builtin", target="echo")

    result = await invoke_tool(tool, {"x": 1}, preflight_client=AllowBuiltinToolPreflightClient())

    assert result.status == "ok"
    assert result.output == {"input": {"x": 1}}


class AllowAllPreflight(ToolPreflightClient):
    def check(self, tool, input_data):
        return ToolPreflightResult(allowed=True, reason="test", preflight_id="pf-test")


async def test_host_specific_tool_still_requires_executor_after_allowed_preflight():
    tool = ToolSpec(name="custom", kind="mcp", target="tools/custom")

    result = await invoke_tool(tool, {}, preflight_client=AllowAllPreflight())

    assert result.status == "unsupported"
    assert result.preflight_id == "pf-test"


def test_builtin_evaluator_scores_exact_match():
    evaluator = EvaluatorSpec(name="json_match", kind="json_schema")

    result = run_evaluator(evaluator, {"x": 1}, {"x": 1})

    assert result.status == "ok"
    assert result.score == 1.0


def test_unknown_constraint_scope_is_rejected():
    data = make_package_dict()
    data["skills"][0]["constraints"][0]["applies_to"] = ["unknown_scope"]

    with pytest.raises(ValueError, match="unknown constraint scopes"):
        SkillPackage(**data)


def test_package_load_rejects_runtime_ref_to_unknown_tool(tmp_path: Path):
    data = make_package_dict()
    data["runtime"]["disabled_tools"] = ["missing_tool"]
    path = tmp_path / "skill-package.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown tools"):
        load_skill_package(path)
