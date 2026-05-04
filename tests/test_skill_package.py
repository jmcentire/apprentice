from pathlib import Path

import pytest
import yaml

from apprentice.skill_package import (
    SkillPackage,
    compose_skill_packages,
    evaluate_constraint_expression,
    load_skill_package,
    merge_package_data,
    read_path,
)


def make_package_dict() -> dict:
    return {
        "package_id": "example.support",
        "version": 1,
        "skills": [
            {
                "name": "classify_ticket",
                "description": "Classify support tickets.",
                "methods": [
                    {
                        "name": "run",
                        "kind": "http",
                        "target": "/v1/run",
                        "method": "POST",
                    }
                ],
                "inputs": [
                    {
                        "name": "text",
                        "type": "string",
                        "modality": "text",
                    }
                ],
                "outputs": [
                    {
                        "name": "category",
                        "type": "string",
                    }
                ],
                "outcomes": [
                    {
                        "name": "accepted",
                        "score": 1.0,
                    }
                ],
                "constraints": [
                    {
                        "name": "pii_enabled",
                        "kind": "hard",
                        "expression": "pii.enabled == true",
                    }
                ],
                "tools": [
                    {
                        "name": "ticket_lookup",
                        "kind": "http",
                        "target": "https://support.example.local/tickets/{ticket_id}",
                    }
                ],
                "evaluators": [
                    {
                        "name": "exact_category",
                        "kind": "json_schema",
                        "applies_to": ["accepted"],
                    }
                ],
                "artifacts": [
                    {
                        "name": "ticket_text",
                        "modality": "text",
                        "path": "input.text",
                        "required": True,
                    }
                ],
                "actions": [
                    {
                        "name": "propose_classification",
                        "outcomes": ["accepted"],
                        "constraints": ["pii_enabled"],
                    }
                ],
            }
        ],
        "runtime": {
            "name": "default",
            "tool_endpoint_overrides": {
                "ticket_lookup": "https://support.example.internal/tickets/{ticket_id}",
            },
        },
        "event_mappings": [
            {
                "event_type": "ticket.classified",
                "skill": "classify_ticket",
                "input_path": "payload.input",
                "output_path": "payload.output",
            }
        ],
    }


def test_load_skill_package_from_yaml(tmp_path: Path):
    path = tmp_path / "skill-package.yaml"
    path.write_text(yaml.safe_dump(make_package_dict()), encoding="utf-8")

    package = load_skill_package(path)

    assert package.package_id == "example.support"
    assert package.skill_names == frozenset({"classify_ticket"})
    assert package.mapping_for_event("ticket.classified").skill == "classify_ticket"


def test_rejects_event_mapping_to_unknown_skill():
    data = make_package_dict()
    data["event_mappings"][0]["skill"] = "missing_skill"

    with pytest.raises(ValueError, match="unknown skill"):
        SkillPackage(**data)


def test_rejects_action_references_to_unknown_outcome():
    data = make_package_dict()
    data["skills"][0]["actions"][0]["outcomes"] = ["missing_outcome"]

    with pytest.raises(ValueError, match="unknown outcomes"):
        SkillPackage(**data)


def test_rejects_unsupported_schema_version():
    data = make_package_dict()
    data["schema_version"] = 999

    with pytest.raises(ValueError, match="unsupported skill package schema_version"):
        SkillPackage(**data)


def test_rejects_evaluator_reference_to_unknown_target():
    data = make_package_dict()
    data["skills"][0]["evaluators"][0]["applies_to"] = ["missing"]

    with pytest.raises(ValueError, match="applies to unknown target"):
        SkillPackage(**data)


def test_read_path_handles_nested_dicts_and_lists():
    data = {"payload": {"items": [{"text": "hello"}]}}

    assert read_path(data, "payload.items.0.text") == "hello"
    assert read_path(data, "payload.items.1.text") is None


def test_skill_for_resolves_tenant_qualified_names():
    package = SkillPackage(**make_package_dict())

    assert package.skill_for("classify_ticket").name == "classify_ticket"
    assert package.skill_for("tenant-1:classify_ticket").name == "classify_ticket"
    assert package.skill_for("tenant-1:missing") is None


def test_safe_constraint_expression_evaluator():
    context = {
        "input": {"text": "card declined", "priority": "high"},
        "pii": {"enabled": True},
    }

    assert evaluate_constraint_expression("not_empty(input.text)", context)[0] is True
    assert evaluate_constraint_expression("pii.enabled == true", context)[0] is True
    assert evaluate_constraint_expression("input.priority in ['high', 'urgent']", context)[0] is True
    assert evaluate_constraint_expression("exists(input.customer_id)", context)[0] is False
    assert evaluate_constraint_expression("__import__('os').system('x')", context)[0] is False


def test_constraint_check_blocks_hard_violations():
    package = SkillPackage(**make_package_dict())

    result = package.check_constraints(
        "tenant-1:classify_ticket",
        action_name="propose_classification",
        context={"pii": {"enabled": False}},
    )

    assert result.allowed is False
    assert result.violations[0].name == "pii_enabled"


def test_runtime_resolves_tool_endpoint_overrides():
    package = SkillPackage(**make_package_dict())

    tools = package.resolved_tools("classify_ticket")

    assert len(tools) == 1
    assert tools[0].name == "ticket_lookup"
    assert tools[0].target == "https://support.example.internal/tickets/{ticket_id}"


def test_merge_package_data_overlays_named_skill_members():
    base = make_package_dict()
    overlay = {
        "package_id": "example.support",
        "runtime": {"disabled_tools": ["ticket_lookup"]},
        "skills": [
            {
                "name": "classify_ticket",
                "constraints": [
                    {
                        "name": "has_ticket_text",
                        "kind": "hard",
                        "expression": "not_empty(input.text)",
                    }
                ],
                "actions": [
                    {
                        "name": "propose_classification",
                        "constraints": ["pii_enabled", "has_ticket_text"],
                    }
                ],
            }
        ],
    }

    merged = merge_package_data(base, overlay)
    package = SkillPackage(**merged)

    skill = package.skill_for("classify_ticket")
    assert {c.name for c in skill.constraints} == {"pii_enabled", "has_ticket_text"}
    assert skill.action_for("propose_classification").constraints == ["pii_enabled", "has_ticket_text"]
    assert package.resolved_tools("classify_ticket") == []


def test_load_skill_package_applies_overlay_file(tmp_path: Path):
    base_path = tmp_path / "skill-package.yaml"
    overlay_path = tmp_path / "prod.yaml"
    base_path.write_text(yaml.safe_dump(make_package_dict()), encoding="utf-8")
    overlay_path.write_text(
        yaml.safe_dump(
            {
                "package_id": "example.support",
                "runtime": {
                    "tool_endpoint_overrides": {
                        "ticket_lookup": "https://prod.example/tickets/{ticket_id}",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    package = load_skill_package(base_path, [overlay_path])

    assert package.resolved_tools("classify_ticket")[0].target == "https://prod.example/tickets/{ticket_id}"


def test_compose_skill_packages_rejects_duplicate_skill_names():
    first = SkillPackage(**make_package_dict())
    second = SkillPackage(**make_package_dict())

    with pytest.raises(ValueError, match="duplicate skill"):
        compose_skill_packages([first, second])


def test_compose_skill_packages_preserves_runtime_bindings():
    first = SkillPackage(**make_package_dict())
    second_data = make_package_dict()
    second_data["package_id"] = "example.billing"
    second_data["skills"][0]["name"] = "classify_invoice"
    second_data["skills"][0]["tools"][0]["name"] = "invoice_lookup"
    second_data["skills"][0]["actions"][0]["name"] = "propose_invoice_classification"
    second_data["event_mappings"] = []
    second_data["runtime"] = {
        "tool_endpoint_overrides": {
            "invoice_lookup": "https://billing.example/invoices/{invoice_id}",
        }
    }
    second = SkillPackage(**second_data)

    package = compose_skill_packages([first, second])

    assert package.resolved_tools("classify_ticket")[0].target == "https://support.example.internal/tickets/{ticket_id}"
    assert package.resolved_tools("classify_invoice")[0].target == "https://billing.example/invoices/{invoice_id}"
