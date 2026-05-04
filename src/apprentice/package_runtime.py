"""Runtime helpers for external Apprentice skill packages.

This module is intentionally small and host-neutral: it validates package
runtime wiring, resolves credentials without leaking secrets, and executes only
the safe tool/evaluator subset that Apprentice can support directly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from apprentice.skill_package import (
    KNOWN_CONSTRAINT_SCOPES,
    EvaluatorSpec,
    SkillPackage,
    ToolSpec,
    read_path,
)


class PackageDiagnostic(BaseModel):
    """One package validation diagnostic."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    level: str = Field(pattern=r"^(error|warning|info)$")
    path: str
    message: str


class PackageValidationReport(BaseModel):
    """Validation report for a package and runtime environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ok: bool
    package_id: str
    version: int
    fingerprint: str
    diagnostics: list[PackageDiagnostic] = Field(default_factory=list)


class CredentialResolution(BaseModel):
    """Redacted credential resolution status."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    auth_ref: str
    source: str
    configured: bool
    redacted: str = ""


class ToolInvocationResult(BaseModel):
    """Result of invoking a package tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(pattern=r"^(ok|error|unsupported|blocked)$")
    tool: str
    output: Any = None
    error: str = ""
    duration_ms: float = 0.0
    preflight_id: str = ""


class ToolPreflightResult(BaseModel):
    """Decision from a preflight gate before tool execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reason: str = ""
    preflight_id: str = ""


class ToolPreflightClient:
    """Default fail-closed package tool preflight client."""

    def check(self, tool: ToolSpec, input_data: Mapping[str, Any]) -> ToolPreflightResult:
        return ToolPreflightResult(
            allowed=False,
            reason=f"no preflight client configured for tool '{tool.name}'",
        )


class AllowBuiltinToolPreflightClient(ToolPreflightClient):
    """Permit inert builtin tools while failing closed for external tools."""

    def check(self, tool: ToolSpec, input_data: Mapping[str, Any]) -> ToolPreflightResult:
        if tool.kind == "builtin":
            return ToolPreflightResult(allowed=True, reason="builtin tool", preflight_id="builtin")
        return super().check(tool, input_data)


class EvaluatorRunResult(BaseModel):
    """Result of running a package evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(pattern=r"^(ok|error|unsupported)$")
    evaluator: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    details: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


class PackageSkillStatus(BaseModel):
    """Lightweight state for dynamically materialized package skills."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    phase: str
    decisions: int = Field(ge=0)
    budget_remaining: float = 0.0


class PackageRuntimeState:
    """EMA-based state for package skills not present in static task config."""

    def __init__(self, *, alpha: float = 0.1, promotion_threshold: float = 0.85, min_decisions: int = 50):
        self._alpha = alpha
        self._promotion_threshold = promotion_threshold
        self._min_decisions = min_decisions
        self._scores: dict[str, tuple[int, float]] = {}

    def record_feedback(self, task_name: str, score: float) -> PackageSkillStatus:
        if not task_name:
            raise ValueError("task_name is required")
        decisions, current = self._scores.get(task_name, (0, 0.0))
        next_decisions = decisions + 1
        next_score = score if decisions == 0 else current * (1 - self._alpha) + score * self._alpha
        self._scores[task_name] = (next_decisions, max(0.0, min(1.0, next_score)))
        return self.status(task_name)

    def status(self, task_name: str) -> PackageSkillStatus:
        decisions, score = self._scores.get(task_name, (0, 0.0))
        return PackageSkillStatus(
            task_name=task_name,
            confidence=score,
            phase=self._phase_for(decisions, score),
            decisions=decisions,
            budget_remaining=0.0,
        )

    def all_statuses(self) -> list[PackageSkillStatus]:
        return [self.status(name) for name in sorted(self._scores)]

    def _phase_for(self, decisions: int, score: float) -> str:
        if decisions == 0:
            return "phase_0"
        if decisions < self._min_decisions:
            return "phase_1"
        if score >= self._promotion_threshold:
            return "phase_3"
        return "phase_2"


class PackageRegistryStore:
    """Persist active package registry metadata for diagnostics and migrations."""

    def __init__(self, path: Path):
        self.path = path

    def publish(self, package: SkillPackage, *, environment: Optional[str] = None, overlays: Optional[list[str]] = None) -> dict[str, Any]:
        report = validate_package_runtime(package, environment=environment)
        record = {
            "package_id": package.package_id,
            "version": package.version,
            "schema_version": package.schema_version,
            "fingerprint": report.fingerprint,
            "environment": environment or package.runtime.name,
            "overlays": overlays or [],
            "loaded_at": datetime.now(timezone.utc).isoformat(),
            "valid": report.ok,
            "diagnostics": [diagnostic.model_dump() for diagnostic in report.diagnostics],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.path)
        return record

    def load(self) -> Optional[dict[str, Any]]:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CredentialResolver:
    """Resolve package auth refs from the environment."""

    env: Mapping[str, str] = field(default_factory=lambda: os.environ)

    def resolve(self, auth_ref: Optional[str]) -> Optional[str]:
        if not auth_ref:
            return None
        env_name = auth_ref[4:] if auth_ref.startswith("env:") else auth_ref
        return self.env.get(env_name)

    def diagnose(self, auth_ref: str) -> CredentialResolution:
        env_name = auth_ref[4:] if auth_ref.startswith("env:") else auth_ref
        value = self.env.get(env_name)
        return CredentialResolution(
            auth_ref=auth_ref,
            source=f"env:{env_name}",
            configured=value is not None,
            redacted=redact_secret(value) if value else "",
        )


def package_fingerprint(package: SkillPackage) -> str:
    """Stable hash of a fully composed package."""
    payload = json.dumps(package.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_package_runtime(
    package: SkillPackage,
    *,
    environment: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
) -> PackageValidationReport:
    """Validate package runtime wiring beyond Pydantic schema checks."""
    diagnostics: list[PackageDiagnostic] = []
    resolver = CredentialResolver(env or os.environ)
    runtime = package.runtime_for(environment)

    known_skills = {skill.name for skill in package.skills}
    known_tools = {tool.name for skill in package.skills for tool in skill.tools}
    known_evaluators = {evaluator.name for skill in package.skills for evaluator in skill.evaluators}

    for name in runtime.enabled_skills:
        if name not in known_skills:
            diagnostics.append(_diag("error", f"runtime.enabled_skills.{name}", "unknown skill"))
    for name in runtime.disabled_tools:
        if name not in known_tools:
            diagnostics.append(_diag("warning", f"runtime.disabled_tools.{name}", "unknown tool"))
    for name in runtime.tool_endpoint_overrides:
        if name not in known_tools:
            diagnostics.append(_diag("error", f"runtime.tool_endpoint_overrides.{name}", "unknown tool"))
    for name, auth_ref in runtime.auth_refs.items():
        if name not in known_tools and name not in known_evaluators:
            diagnostics.append(_diag("warning", f"runtime.auth_refs.{name}", "unknown tool or evaluator"))
        resolution = resolver.diagnose(auth_ref)
        if not resolution.configured:
            diagnostics.append(_diag("warning", f"runtime.auth_refs.{name}", f"{resolution.source} is not set"))

    for skill in package.skills:
        for constraint in skill.constraints:
            unknown_scopes = [scope for scope in constraint.applies_to if scope not in KNOWN_CONSTRAINT_SCOPES]
            for scope in unknown_scopes:
                diagnostics.append(
                    _diag("error", f"skills.{skill.name}.constraints.{constraint.name}.applies_to", f"unknown scope '{scope}'")
                )
        for tool in skill.tools:
            if tool.auth_ref and resolver.resolve(tool.auth_ref) is None:
                diagnostics.append(_diag("warning", f"skills.{skill.name}.tools.{tool.name}.auth_ref", "credential is not set"))

    return PackageValidationReport(
        ok=not any(d.level == "error" for d in diagnostics),
        package_id=package.package_id,
        version=package.version,
        fingerprint=package_fingerprint(package),
        diagnostics=diagnostics,
    )


def diff_packages(old: SkillPackage, new: SkillPackage) -> dict[str, Any]:
    """Return a migration-oriented diff between two packages."""
    old_skills = {skill.name: skill for skill in old.skills}
    new_skills = {skill.name: skill for skill in new.skills}
    removed = sorted(set(old_skills) - set(new_skills))
    added = sorted(set(new_skills) - set(old_skills))
    changed = sorted(
        name
        for name in set(old_skills) & set(new_skills)
        if old_skills[name].model_dump(mode="json") != new_skills[name].model_dump(mode="json")
    )
    warnings: list[str] = []
    safe: list[str] = []
    for name in changed:
        old_skill = old_skills[name]
        new_skill = new_skills[name]
        old_actions = {action.name for action in old_skill.actions}
        new_actions = {action.name for action in new_skill.actions}
        old_outcomes = {outcome.name for outcome in old_skill.outcomes}
        new_outcomes = {outcome.name for outcome in new_skill.outcomes}
        old_required_inputs = {field.name for field in old_skill.inputs if field.required}
        new_required_inputs = {field.name for field in new_skill.inputs if field.required}
        removed_parts = (
            sorted(old_actions - new_actions)
            + sorted(old_outcomes - new_outcomes)
            + sorted(old_required_inputs - new_required_inputs)
        )
        added_required_inputs = sorted(new_required_inputs - old_required_inputs)
        if removed_parts or added_required_inputs:
            warnings.append(name)
        else:
            safe.append(name)

    breaking = sorted(set(removed) | set(warnings))
    return {
        "from": {"package_id": old.package_id, "version": old.version, "fingerprint": package_fingerprint(old)},
        "to": {"package_id": new.package_id, "version": new.version, "fingerprint": package_fingerprint(new)},
        "skills_added": added,
        "skills_removed": removed,
        "skills_changed": changed,
        "breaking_changes": breaking,
        "warning_changes": warnings,
        "safe_changes": safe + added,
        "breaking": bool(breaking),
    }


async def invoke_tool(
    tool: ToolSpec,
    input_data: Mapping[str, Any],
    *,
    credential_resolver: Optional[CredentialResolver] = None,
    preflight_client: Optional[ToolPreflightClient] = None,
) -> ToolInvocationResult:
    """Invoke a package tool if it is supported by the built-in runtime."""
    preflight = (preflight_client or AllowBuiltinToolPreflightClient()).check(tool, input_data)
    if not preflight.allowed:
        return ToolInvocationResult(
            status="blocked",
            tool=tool.name,
            error=preflight.reason,
            preflight_id=preflight.preflight_id,
        )
    if tool.kind == "builtin":
        return ToolInvocationResult(
            status="ok",
            tool=tool.name,
            output={"input": dict(input_data)},
            preflight_id=preflight.preflight_id,
        )
    if tool.kind != "http":
        return ToolInvocationResult(
            status="unsupported",
            tool=tool.name,
            error=f"tool kind '{tool.kind}' requires a host-specific executor",
            preflight_id=preflight.preflight_id,
        )

    started = asyncio.get_running_loop().time()
    try:
        output = await asyncio.to_thread(_invoke_http_tool, tool, input_data, credential_resolver or CredentialResolver())
        duration = (asyncio.get_running_loop().time() - started) * 1000
        return ToolInvocationResult(status="ok", tool=tool.name, output=output, duration_ms=duration, preflight_id=preflight.preflight_id)
    except Exception as exc:
        duration = (asyncio.get_running_loop().time() - started) * 1000
        return ToolInvocationResult(status="error", tool=tool.name, error=str(exc), duration_ms=duration, preflight_id=preflight.preflight_id)


def run_evaluator(evaluator: EvaluatorSpec, candidate: Any, reference: Any) -> EvaluatorRunResult:
    """Run built-in package evaluator kinds."""
    if evaluator.kind in {"builtin", "json_schema"}:
        score = 1.0 if candidate == reference else 0.0
        return EvaluatorRunResult(
            status="ok",
            evaluator=evaluator.name,
            score=score,
            details={"kind": evaluator.kind, "match": candidate == reference},
        )
    if evaluator.kind == "regex":
        pattern = str(evaluator.config.get("pattern") or evaluator.target)
        if not pattern:
            return EvaluatorRunResult(status="error", evaluator=evaluator.name, error="regex evaluator missing pattern")
        text = json.dumps(candidate, sort_keys=True) if not isinstance(candidate, str) else candidate
        matched = re.search(pattern, text) is not None
        return EvaluatorRunResult(status="ok", evaluator=evaluator.name, score=1.0 if matched else 0.0, details={"matched": matched})
    return EvaluatorRunResult(
        status="unsupported",
        evaluator=evaluator.name,
        error=f"evaluator kind '{evaluator.kind}' requires a host-specific executor",
    )


def redact_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:2]}***{value[-2:]}"


def _invoke_http_tool(tool: ToolSpec, input_data: Mapping[str, Any], resolver: CredentialResolver) -> Any:
    url = _format_target(tool.target, input_data)
    if not url.startswith(("http://", "https://")):
        raise ValueError("http tool target must start with http:// or https://")
    method = "POST"
    method = method.upper()
    payload = json.dumps(dict(input_data)).encode("utf-8")
    headers = {"Accept": "application/json"}
    data = payload
    if method == "GET":
        data = None
    else:
        headers["Content-Type"] = "application/json"
    credential = resolver.resolve(tool.auth_ref)
    if credential:
        headers["Authorization"] = f"Bearer {credential}"
    request = urllib.request.Request(url=url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=tool.timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"http tool returned {exc.code}") from exc
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"text": body}


def _format_target(target: str, input_data: Mapping[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        value = read_path(input_data, match.group(1))
        if value is None:
            raise ValueError(f"missing template value '{match.group(1)}'")
        return urllib.parse.quote(str(value), safe="")

    return re.sub(r"\{([A-Za-z0-9_.:-]+)\}", replace, target)


def _diag(level: str, path: str, message: str) -> PackageDiagnostic:
    return PackageDiagnostic(level=level, path=path, message=message)
