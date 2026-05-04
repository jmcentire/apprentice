"""Domain-neutral skill package configuration.

Skill packages let host applications describe their own actions,
outcomes, tools, constraints, and event mappings outside this repo.
Apprentice loads the package at launch and treats it as the vocabulary
for learning signals.
"""

from pathlib import Path
import ast
import copy
import re
from typing import Any, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JsonObject = dict[str, Any]
SUPPORTED_SCHEMA_VERSION = 1
KNOWN_CONSTRAINT_SCOPES = frozenset(
    {
        "run",
        "training_data",
        "tool_call",
        "feedback",
        "promotion",
        "autonomous_ship",
        "evaluation",
        "event",
    }
)


class ConstraintViolation(BaseModel):
    """A package constraint that failed or could not be evaluated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: str
    expression: str
    reason: str


class ConstraintCheckResult(BaseModel):
    """Result of checking package constraints against a request context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    violations: list[ConstraintViolation] = Field(default_factory=list)


class CompatibilitySpec(BaseModel):
    """Version contract between Apprentice and a package."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_apprentice_version: Optional[str] = None
    max_apprentice_version: Optional[str] = None
    package_api_version: int = Field(default=1, ge=1)


class RuntimeBindingSpec(BaseModel):
    """Deploy-time runtime binding selected by config or environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="default", pattern=r"^[a-z][a-z0-9_-]*$")
    tool_endpoint_overrides: dict[str, str] = Field(default_factory=dict)
    auth_refs: dict[str, str] = Field(default_factory=dict)
    enabled_skills: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)
    feature_flags: dict[str, bool] = Field(default_factory=dict)


class InvocationSpec(BaseModel):
    """How a host application evokes a skill or method."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    kind: str = Field(pattern=r"^(http|python|cli|mcp|manual)$")
    target: str = Field(min_length=1)
    method: Optional[str] = None
    timeout_seconds: int = Field(default=30, ge=1, le=3600)
    auth_ref: Optional[str] = None
    constraints: list[str] = Field(default_factory=list)


class ParameterSpec(BaseModel):
    """A named input or output parameter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    type: str = Field(min_length=1)
    required: bool = True
    description: str = ""
    modality: str = Field(default="json", pattern=r"^(json|text|image|audio|video|file|embedding)$")


class ConstraintSpec(BaseModel):
    """A hard or soft rule the host wants Apprentice to respect."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    kind: str = Field(default="hard", pattern=r"^(hard|soft)$")
    applies_to: list[str] = Field(default_factory=list)
    expression: str = Field(min_length=1)
    description: str = ""

    @field_validator("applies_to")
    @classmethod
    def validate_applies_to(cls, scopes: list[str]) -> list[str]:
        unknown = [scope for scope in scopes if scope not in KNOWN_CONSTRAINT_SCOPES]
        if unknown:
            raise ValueError(f"unknown constraint scopes: {unknown}")
        return scopes


class OutcomeSpec(BaseModel):
    """A learnable success, failure, or quality outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = ""
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    success_when: str = Field(default="true")


class ActionSpec(BaseModel):
    """An action proposal or completed action within a skill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    description: str = ""
    parameters: list[ParameterSpec] = Field(default_factory=list)
    outcomes: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    reversible: bool = True


class ToolSpec(BaseModel):
    """A tool available to the teacher, local model, or evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    kind: str = Field(pattern=r"^(http|python|cli|mcp|builtin)$")
    target: str = Field(min_length=1)
    enabled: bool = True
    auth_ref: Optional[str] = None
    timeout_seconds: int = Field(default=30, ge=1, le=3600)
    rate_limit_per_minute: Optional[int] = Field(default=None, ge=1, le=60000)
    input_schema: JsonObject = Field(default_factory=dict)
    output_schema: JsonObject = Field(default_factory=dict)
    description: str = ""


class EvaluatorSpec(BaseModel):
    """Evaluator binding for a skill, action, or outcome."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    kind: str = Field(pattern=r"^(builtin|python|http|mcp|llm_judge|semantic_similarity|json_schema|regex)$")
    target: str = Field(default="")
    applies_to: list[str] = Field(default_factory=list)
    config: JsonObject = Field(default_factory=dict)
    required: bool = True


class ArtifactSpec(BaseModel):
    """Named multimodal artifact attached to inputs, outputs, or feedback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    modality: str = Field(pattern=r"^(json|text|image|audio|video|file|embedding)$")
    path: str = Field(min_length=1)
    required: bool = False
    description: str = ""


class FeedbackSignalSpec(BaseModel):
    """A feedback signal and its normalized learning score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    score: float = Field(ge=0.0, le=1.0)
    description: str = ""


class EventMappingSpec(BaseModel):
    """Map a host event into an Apprentice skill-learning record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: str = Field(min_length=1)
    skill: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    input_path: str = Field(default="payload")
    output_path: Optional[str] = None
    feedback_path: Optional[str] = None
    subject_path: Optional[str] = None
    action: Optional[str] = None
    outcome_path: Optional[str] = None


class SkillSpec(BaseModel):
    """Full host-defined description of one learnable skill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = ""
    methods: list[InvocationSpec] = Field(default_factory=list)
    inputs: list[ParameterSpec] = Field(default_factory=list)
    outputs: list[ParameterSpec] = Field(default_factory=list)
    actions: list[ActionSpec] = Field(default_factory=list)
    outcomes: list[OutcomeSpec] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    tools: list[ToolSpec] = Field(default_factory=list)
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)
    artifacts: list[ArtifactSpec] = Field(default_factory=list)
    feedback_signals: list[FeedbackSignalSpec] = Field(default_factory=list)

    def method_for(self, name: str) -> Optional[InvocationSpec]:
        for method in self.methods:
            if method.name == name:
                return method
        return None

    def action_for(self, name: str) -> Optional[ActionSpec]:
        for action in self.actions:
            if action.name == name:
                return action
        return None

    def constraints_for_action(self, action_name: Optional[str]) -> list[ConstraintSpec]:
        if not action_name:
            return self.constraints
        action = self.action_for(action_name)
        if action is None:
            return []
        wanted = set(action.constraints)
        return [constraint for constraint in self.constraints if constraint.name in wanted]


class SkillPackage(BaseModel):
    """A domain package loaded from an external YAML file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    package_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    schema_version: int = Field(default=SUPPORTED_SCHEMA_VERSION, ge=1)
    version: int = Field(default=1, ge=1)
    description: str = ""
    compatibility: CompatibilitySpec = Field(default_factory=CompatibilitySpec)
    runtime: RuntimeBindingSpec = Field(default_factory=RuntimeBindingSpec)
    environments: dict[str, RuntimeBindingSpec] = Field(default_factory=dict)
    skills: list[SkillSpec] = Field(min_length=1)
    event_mappings: list[EventMappingSpec] = Field(default_factory=list)

    @field_validator("skills")
    @classmethod
    def validate_unique_skills(cls, skills: list[SkillSpec]) -> list[SkillSpec]:
        names = [s.name for s in skills]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate skill names: {duplicates}")
        return skills

    @model_validator(mode="after")
    def validate_references(self) -> "SkillPackage":
        if self.schema_version > SUPPORTED_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported skill package schema_version {self.schema_version}; "
                f"this Apprentice supports <= {SUPPORTED_SCHEMA_VERSION}"
            )
        skill_names = {s.name for s in self.skills}
        for mapping in self.event_mappings:
            if mapping.skill not in skill_names:
                raise ValueError(
                    f"event mapping for '{mapping.event_type}' references unknown skill '{mapping.skill}'"
                )
            if mapping.action:
                skill = next(s for s in self.skills if s.name == mapping.skill)
                if skill.action_for(mapping.action) is None:
                    raise ValueError(
                        f"event mapping for '{mapping.event_type}' references unknown action '{mapping.action}'"
                    )
        for skill in self.skills:
            outcome_names = {o.name for o in skill.outcomes}
            constraint_names = {c.name for c in skill.constraints}
            for method in skill.methods:
                missing_constraints = [c for c in method.constraints if c not in constraint_names]
                if missing_constraints:
                    raise ValueError(
                        f"method '{method.name}' references unknown constraints: {missing_constraints}"
                    )
            for action in skill.actions:
                missing_outcomes = [o for o in action.outcomes if o not in outcome_names]
                missing_constraints = [c for c in action.constraints if c not in constraint_names]
                if missing_outcomes:
                    raise ValueError(f"action '{action.name}' references unknown outcomes: {missing_outcomes}")
                if missing_constraints:
                    raise ValueError(
                        f"action '{action.name}' references unknown constraints: {missing_constraints}"
                    )
            for evaluator in skill.evaluators:
                for target in evaluator.applies_to:
                    if (
                        target not in outcome_names
                        and target not in {a.name for a in skill.actions}
                        and target not in {m.name for m in skill.methods}
                    ):
                        raise ValueError(
                            f"evaluator '{evaluator.name}' applies to unknown target '{target}'"
                        )
        self._validate_runtime_references(self.runtime, "runtime")
        for name, runtime in self.environments.items():
            self._validate_runtime_references(runtime, f"environments.{name}")
        return self

    @property
    def skill_names(self) -> frozenset[str]:
        return frozenset(skill.name for skill in self.skills)

    def mapping_for_event(self, event_type: str) -> Optional[EventMappingSpec]:
        for mapping in self.event_mappings:
            if mapping.event_type == event_type:
                return mapping
        return None

    def skill_for(self, name: str) -> Optional[SkillSpec]:
        """Resolve an exact or tenant-qualified skill name.

        Host apps may use tenant-qualified task names such as
        ``tenant-id:send_email_reply`` while the package keeps only the
        generic skill vocabulary.
        """
        for skill in self.skills:
            if skill.name == name:
                return skill
        if ":" in name:
            suffix = name.rsplit(":", 1)[1]
            for skill in self.skills:
                if skill.name == suffix:
                    return skill
        return None

    def check_constraints(
        self,
        skill_name: str,
        *,
        action_name: Optional[str] = None,
        context: Optional[JsonObject] = None,
    ) -> ConstraintCheckResult:
        skill = self.skill_for(skill_name)
        if skill is None:
            violation = ConstraintViolation(
                name="unknown_skill",
                kind="hard",
                expression=skill_name,
                reason=f"skill '{skill_name}' is not defined by package '{self.package_id}'",
            )
            return ConstraintCheckResult(allowed=False, violations=[violation])

        if action_name and skill.action_for(action_name) is None:
            violation = ConstraintViolation(
                name="unknown_action",
                kind="hard",
                expression=action_name,
                reason=f"action '{action_name}' is not defined for skill '{skill.name}'",
            )
            return ConstraintCheckResult(allowed=False, violations=[violation])

        violations: list[ConstraintViolation] = []
        selected = skill.constraints_for_action(action_name)
        method_name = str((context or {}).get("method", ""))
        scope = str((context or {}).get("scope", ""))
        if method_name:
            method = skill.method_for(method_name)
            if method is not None and method.constraints:
                wanted = set(method.constraints)
                selected = [c for c in skill.constraints if c.name in wanted]

        for constraint in selected:
            if constraint.applies_to and scope and scope not in constraint.applies_to:
                continue
            ok, reason = evaluate_constraint_expression(constraint.expression, context or {})
            if not ok:
                violations.append(
                    ConstraintViolation(
                        name=constraint.name,
                        kind=constraint.kind,
                        expression=constraint.expression,
                        reason=reason,
                    )
                )
        return ConstraintCheckResult(
            allowed=not any(v.kind == "hard" for v in violations),
            violations=violations,
        )

    def _validate_runtime_references(self, runtime: RuntimeBindingSpec, path: str) -> None:
        skill_names = {skill.name for skill in self.skills}
        tool_names = {tool.name for skill in self.skills for tool in skill.tools}
        evaluator_names = {evaluator.name for skill in self.skills for evaluator in skill.evaluators}
        missing_skills = [name for name in runtime.enabled_skills if name not in skill_names]
        if missing_skills:
            raise ValueError(f"{path}.enabled_skills references unknown skills: {missing_skills}")
        missing_disabled = [name for name in runtime.disabled_tools if name not in tool_names]
        if missing_disabled:
            raise ValueError(f"{path}.disabled_tools references unknown tools: {missing_disabled}")
        missing_overrides = [name for name in runtime.tool_endpoint_overrides if name not in tool_names]
        if missing_overrides:
            raise ValueError(f"{path}.tool_endpoint_overrides references unknown tools: {missing_overrides}")
        missing_auth = [
            name
            for name in runtime.auth_refs
            if name not in tool_names and name not in evaluator_names
        ]
        if missing_auth:
            raise ValueError(f"{path}.auth_refs references unknown tools or evaluators: {missing_auth}")

    def runtime_for(self, environment: Optional[str] = None) -> RuntimeBindingSpec:
        if environment and environment in self.environments:
            return self.environments[environment]
        return self.runtime

    def resolved_tools(self, skill_name: str, *, environment: Optional[str] = None) -> list[ToolSpec]:
        skill = self.skill_for(skill_name)
        if skill is None:
            return []
        runtime = self.runtime_for(environment)
        disabled = set(runtime.disabled_tools)
        overrides = runtime.tool_endpoint_overrides
        tools: list[ToolSpec] = []
        for tool in skill.tools:
            if not tool.enabled or tool.name in disabled:
                continue
            update: dict[str, Any] = {}
            if tool.name in overrides:
                update["target"] = overrides[tool.name]
            if tool.name in runtime.auth_refs:
                update["auth_ref"] = runtime.auth_refs[tool.name]
            tools.append(tool.model_copy(update=update))
        return tools


def load_skill_package(
    path: Path,
    overlay_paths: Optional[list[Path]] = None,
    *,
    environment: Optional[str] = None,
) -> SkillPackage:
    """Load and validate a skill package YAML file."""
    data = _read_package_yaml(path)
    for overlay_path in overlay_paths or []:
        overlay_data = _read_package_yaml(overlay_path)
        data = merge_package_data(data, overlay_data)
    package = SkillPackage(**data)
    if environment and environment not in package.environments and environment != package.runtime.name:
        raise ValueError(f"skill package environment '{environment}' is not defined")
    return package


def compose_skill_packages(packages: list[SkillPackage]) -> SkillPackage:
    """Compose independent packages into one runtime registry package."""
    if not packages:
        raise ValueError("at least one skill package is required")
    if len(packages) == 1:
        return packages[0]

    data: dict[str, Any] = {
        "package_id": "apprentice.bundle",
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "version": max(package.version for package in packages),
        "description": "Composed Apprentice skill package registry.",
        "runtime": {
            "name": "default",
            "tool_endpoint_overrides": {},
            "auth_refs": {},
            "enabled_skills": [],
            "disabled_tools": [],
            "feature_flags": {},
        },
        "environments": {},
        "skills": [],
        "event_mappings": [],
    }
    seen_skills: set[str] = set()
    seen_events: set[str] = set()
    for package in packages:
        for skill in package.skills:
            if skill.name in seen_skills:
                raise ValueError(f"duplicate skill '{skill.name}' across skill packages")
            seen_skills.add(skill.name)
            data["skills"].append(skill.model_dump())
        for mapping in package.event_mappings:
            if mapping.event_type in seen_events:
                raise ValueError(f"duplicate event mapping '{mapping.event_type}' across skill packages")
            seen_events.add(mapping.event_type)
            data["event_mappings"].append(mapping.model_dump())
        data["runtime"] = _merge_runtime_data(data["runtime"], package.runtime.model_dump())
        for env_name, runtime in package.environments.items():
            existing = data["environments"].get(env_name)
            data["environments"][env_name] = (
                _merge_runtime_data(existing, runtime.model_dump())
                if existing is not None
                else runtime.model_dump()
            )
    return SkillPackage(**data)


def merge_package_data(base: JsonObject, overlay: JsonObject) -> JsonObject:
    """Merge an overlay package YAML object onto a base package object."""
    result = copy.deepcopy(base)
    overlay_copy = copy.deepcopy(overlay)

    package_id = overlay_copy.pop("package_id", None)
    if package_id is not None and package_id != result.get("package_id"):
        raise ValueError("skill package overlays must use the same package_id as the base package")

    for key in ("skills", "event_mappings"):
        if key in overlay_copy:
            result[key] = _merge_named_list(
                result.get(key, []),
                overlay_copy.pop(key) or [],
                "event_type" if key == "event_mappings" else "name",
                merge_skill_members=key == "skills",
            )

    for key, value in overlay_copy.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def read_path(data: Any, path: str) -> Any:
    """Read a dotted path from nested dict/list data."""
    current = data
    if path == "":
        return current
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit():
            index = int(part)
            if index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def _read_package_yaml(path: Path) -> JsonObject:
    if not path.exists():
        raise FileNotFoundError(f"Skill package not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Skill package root must be a YAML object: {path}")
    return data


def _merge_named_list(
    base: list[Any],
    overlay: list[Any],
    key: str,
    *,
    merge_skill_members: bool = False,
) -> list[Any]:
    result = copy.deepcopy(base)
    index = {
        item.get(key): i
        for i, item in enumerate(result)
        if isinstance(item, dict) and item.get(key) is not None
    }
    for item in overlay:
        if not isinstance(item, dict) or item.get(key) not in index:
            result.append(item)
            continue
        existing = result[index[item[key]]]
        result[index[item[key]]] = (
            _merge_skill_like(existing, item)
            if merge_skill_members
            else _deep_merge_dict(existing, item)
        )
    return result


def _merge_skill_like(base: JsonObject, overlay: JsonObject) -> JsonObject:
    result = copy.deepcopy(base)
    overlay_copy = copy.deepcopy(overlay)
    for key in (
        "methods",
        "inputs",
        "outputs",
        "actions",
        "outcomes",
        "constraints",
        "tools",
        "evaluators",
        "artifacts",
        "feedback_signals",
    ):
        if key in overlay_copy:
            result[key] = _merge_named_list(result.get(key, []), overlay_copy.pop(key) or [], "name")
    return _deep_merge_dict(result, overlay_copy)


def _deep_merge_dict(base: JsonObject, overlay: JsonObject) -> JsonObject:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _merge_runtime_data(base: JsonObject, overlay: JsonObject) -> JsonObject:
    result = copy.deepcopy(base)
    for key in ("tool_endpoint_overrides", "auth_refs", "feature_flags"):
        result[key] = {**result.get(key, {}), **overlay.get(key, {})}
    for key in ("enabled_skills", "disabled_tools"):
        result[key] = sorted(set(result.get(key, [])) | set(overlay.get(key, [])))
    if overlay.get("name") and result.get("name") == "default":
        result["name"] = overlay["name"]
    return result


def evaluate_constraint_expression(expression: str, context: JsonObject) -> tuple[bool, str]:
    """Evaluate the safe subset of package constraint expressions.

    Supported expressions are intentionally small:
    - ``true`` / ``false``
    - ``exists(path)``
    - ``not_empty(path)``
    - ``path == literal`` / ``path != literal``
    - ``path in [literal, ...]``
    """
    expr = expression.strip()
    if expr == "true":
        return True, ""
    if expr == "false":
        return False, "expression is false"

    match = re.fullmatch(r"(exists|not_empty)\(([A-Za-z0-9_.:-]+)\)", expr)
    if match:
        op, path = match.groups()
        value = read_path(context, path)
        if op == "exists":
            return (value is not None, f"{path} does not exist")
        return (_is_not_empty(value), f"{path} is empty")

    match = re.fullmatch(r"([A-Za-z0-9_.:-]+)\s*(==|!=)\s*(.+)", expr)
    if match:
        path, op, raw_expected = match.groups()
        actual = read_path(context, path)
        expected = _parse_literal(raw_expected)
        ok = actual == expected if op == "==" else actual != expected
        return ok, f"{path} was {actual!r}, expected {op} {expected!r}"

    match = re.fullmatch(r"([A-Za-z0-9_.:-]+)\s+in\s+(\[.*\])", expr)
    if match:
        path, raw_options = match.groups()
        actual = read_path(context, path)
        options = _parse_literal(raw_options)
        if not isinstance(options, list):
            return False, "right side of in expression must be a list"
        return actual in options, f"{path} was {actual!r}, expected one of {options!r}"

    return False, f"unsupported constraint expression: {expression}"


def _parse_literal(value: str) -> Any:
    normalized = value.strip()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if normalized == "null":
        return None
    try:
        return ast.literal_eval(normalized)
    except (SyntaxError, ValueError):
        return normalized.strip("'\"")


def _is_not_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True
