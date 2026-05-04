"""HTTP Daemon for Apprentice.

Bare asyncio HTTP server with background training pipeline loop.
No external framework dependencies.

Security modes:
  none      — No authentication (use behind firewall/VPC only)
  api-key   — Static API key via Authorization: Bearer <key> header
  jwt       — RS256/HS256 JWT verification via Authorization: Bearer <token>
  hmac      — HMAC-SHA256 request signing via X-Signature header

Additional security:
  --tls-cert / --tls-key  — Enable HTTPS
  --allowed-ips           — IP allowlist (comma-separated CIDRs or addresses)
  --host 127.0.0.1        — Default bind to loopback (safe default)
"""

import asyncio
import hashlib
import hmac as hmac_mod
import ipaddress
import json
import os
import signal
import ssl
import sys
from datetime import datetime, timezone
from typing import Any, Optional


# ── Security ──────────────────────────────────────────────────────────────


class SecurityConfig:
    """Runtime security configuration for the HTTP server."""

    def __init__(
        self,
        auth_mode: str = "none",
        api_key: str = "",
        jwt_secret: str = "",
        jwt_algorithm: str = "HS256",
        hmac_secret: str = "",
        tls_cert: str = "",
        tls_key: str = "",
        allowed_ips: Optional[list[str]] = None,
    ):
        self.auth_mode = auth_mode
        self.api_key = api_key
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.hmac_secret = hmac_secret
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self.allowed_ips = allowed_ips or []

        # Parse allowed IPs into network objects for CIDR matching
        self._allowed_networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for ip_str in self.allowed_ips:
            try:
                self._allowed_networks.append(ipaddress.ip_network(ip_str, strict=False))
            except ValueError:
                print(f"[security] Warning: invalid IP/CIDR '{ip_str}', skipping", file=sys.stderr)

    def check_ip(self, client_ip: str) -> bool:
        """Return True if client IP is allowed (or no allowlist configured)."""
        if not self._allowed_networks:
            return True
        try:
            addr = ipaddress.ip_address(client_ip)
            return any(addr in net for net in self._allowed_networks)
        except ValueError:
            return False

    def check_auth(self, headers: dict[str, str], method: str, path: str, body: str) -> tuple[bool, str]:
        """Verify authentication. Returns (allowed, reason)."""
        if self.auth_mode == "none":
            return True, ""

        if self.auth_mode == "api-key":
            return self._check_api_key(headers)

        if self.auth_mode == "jwt":
            return self._check_jwt(headers)

        if self.auth_mode == "hmac":
            return self._check_hmac(headers, method, path, body)

        return False, f"Unknown auth mode: {self.auth_mode}"

    def _check_api_key(self, headers: dict[str, str]) -> tuple[bool, str]:
        auth = headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return False, "Missing Authorization: Bearer <key> header"
        token = auth[7:]
        if not hmac_mod.compare_digest(token, self.api_key):
            return False, "Invalid API key"
        return True, ""

    def _check_jwt(self, headers: dict[str, str]) -> tuple[bool, str]:
        auth = headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return False, "Missing Authorization: Bearer <token> header"
        token = auth[7:]

        # Minimal JWT verification (HS256 only, no external deps)
        # For RS256 or full JWT, users should put a reverse proxy in front
        import base64

        parts = token.split(".")
        if len(parts) != 3:
            return False, "Malformed JWT"

        try:
            # Verify signature
            signing_input = f"{parts[0]}.{parts[1]}".encode()
            signature = base64.urlsafe_b64decode(parts[2] + "==")
            expected = hmac_mod.new(
                self.jwt_secret.encode(), signing_input, hashlib.sha256
            ).digest()
            if not hmac_mod.compare_digest(signature, expected):
                return False, "Invalid JWT signature"

            # Decode payload and check exp
            payload_bytes = base64.urlsafe_b64decode(parts[1] + "==")
            payload = json.loads(payload_bytes)
            if "exp" in payload:
                import time
                if payload["exp"] < time.time():
                    return False, "JWT expired"

        except Exception as e:
            return False, f"JWT verification failed: {e}"

        return True, ""

    def _check_hmac(self, headers: dict[str, str], method: str, path: str, body: str) -> tuple[bool, str]:
        signature = headers.get("x-signature", "")
        if not signature:
            return False, "Missing X-Signature header"

        # Sign: HMAC-SHA256(secret, "METHOD\nPATH\nBODY")
        message = f"{method}\n{path}\n{body}".encode()
        expected = hmac_mod.new(
            self.hmac_secret.encode(), message, hashlib.sha256
        ).hexdigest()
        if not hmac_mod.compare_digest(signature, expected):
            return False, "Invalid HMAC signature"
        return True, ""

    def build_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Build SSL context if TLS is configured, else return None."""
        if not self.tls_cert or not self.tls_key:
            return None
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(self.tls_cert, self.tls_key)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        return ctx


# ── HTTP Server ───────────────────────────────────────────────────────────


class ApprenticeServer:
    """Async HTTP server wrapping an Apprentice instance with configurable security."""

    def __init__(
        self,
        apprentice: Any,
        host: str = "127.0.0.1",
        port: int = 8710,
        pipeline_interval: int = 300,
        security: Optional[SecurityConfig] = None,
    ):
        self._apprentice = apprentice
        self._host = host
        self._port = port
        self._pipeline_interval = pipeline_interval
        self._security = security or SecurityConfig()
        self._server: Optional[asyncio.AbstractServer] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        self._training_in_flight = False

    async def start(self) -> None:
        """Start the HTTP server and background pipeline."""
        ssl_ctx = self._security.build_ssl_context()
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port, ssl=ssl_ctx,
        )
        self._pipeline_task = asyncio.create_task(self._pipeline_loop())

    async def wait_for_shutdown(self) -> None:
        """Block until SIGTERM/SIGINT."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        await self._shutdown_event.wait()
        await self.stop()

    async def stop(self) -> None:
        """Gracefully stop server and pipeline."""
        if self._pipeline_task:
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except asyncio.CancelledError:
                pass

        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # ── HTTP Handler ──────────────────────────────────────────────────────

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle a single HTTP connection with security checks."""
        try:
            # IP allowlist check
            peername = writer.get_extra_info("peername")
            client_ip = peername[0] if peername else "unknown"

            if not self._security.check_ip(client_ip):
                await self._send_response(writer, 403, {"error": "Forbidden: IP not allowed"})
                return

            raw = await asyncio.wait_for(reader.read(65536), timeout=30.0)
            request_text = raw.decode("utf-8", errors="replace")

            # Parse request line
            lines = request_text.split("\r\n")
            if not lines:
                await self._send_response(writer, 400, {"error": "Empty request"})
                return

            request_line = lines[0]
            parts = request_line.split(" ")
            if len(parts) < 2:
                await self._send_response(writer, 400, {"error": "Malformed request"})
                return

            method = parts[0]
            path = parts[1]

            # Parse headers
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if not line:
                    break
                if ": " in line:
                    key, value = line.split(": ", 1)
                    headers[key.lower()] = value

            # Extract body
            body = ""
            if "\r\n\r\n" in request_text:
                body = request_text.split("\r\n\r\n", 1)[1]

            # Auth check (skip for /health — always open for load balancer probes)
            if path != "/health":
                allowed, reason = self._security.check_auth(headers, method, path, body)
                if not allowed:
                    await self._send_response(writer, 401, {"error": f"Unauthorized: {reason}"})
                    return

            # Route
            if path == "/health" and method == "GET":
                await self._handle_health(writer)
            elif path == "/v1/run" and method == "POST":
                await self._handle_run(writer, body)
            elif path == "/v1/status" and method == "GET":
                await self._handle_status(writer)
            elif path == "/v1/report" and method == "GET":
                await self._handle_report(writer)
            elif path == "/v1/events" and method == "POST":
                await self._handle_events(writer, body)
            elif path == "/v1/feedback" and method == "POST":
                await self._handle_feedback(writer, body)
            elif path == "/v1/labeled-examples" and method == "POST":
                await self._handle_labeled_example(writer, body)
            elif path == "/v1/constraints/check" and method == "POST":
                await self._handle_constraint_check(writer, body)
            elif path == "/v1/recommendations" and method == "POST":
                await self._handle_recommendations(writer, body)
            elif path == "/v1/skills" and method == "GET":
                await self._handle_skills(writer)
            elif path == "/v1/skill-package" and method == "GET":
                await self._handle_skill_package(writer)
            elif path == "/v1/package-registry" and method == "GET":
                await self._handle_package_registry(writer)
            elif path == "/v1/tools" and method == "GET":
                await self._handle_tools(writer)
            elif path == "/v1/tools/call" and method == "POST":
                await self._handle_tool_call(writer, body)
            elif path == "/v1/evaluators/run" and method == "POST":
                await self._handle_evaluator_run(writer, body)
            else:
                await self._send_response(writer, 404, {"error": "Not found"})

        except asyncio.TimeoutError:
            await self._send_response(writer, 408, {"error": "Request timeout"})
        except Exception as e:
            await self._send_response(writer, 500, {"error": str(e)})
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_health(self, writer: asyncio.StreamWriter) -> None:
        await self._send_response(writer, 200, {"status": "ok"})

    async def _handle_run(self, writer: asyncio.StreamWriter, body: str) -> None:
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_response(writer, 400, {"error": f"Invalid JSON: {e}"})
            return

        task_name = data.get("task_name") or data.get("skill") or ""
        input_data = data.get("input", {})

        if not task_name:
            await self._send_response(writer, 400, {"error": "task_name is required"})
            return

        package = getattr(self._apprentice, "_skill_package", None)
        if package is not None:
            check = self._package_constraint_check(data, package, scope="run")
            if not check["allowed"]:
                await self._send_response(writer, 422, {
                    "status": "blocked",
                    "package_id": package.package_id,
                    "skill": task_name,
                    "violations": check["violations"],
                })
                return

        try:
            response = await self._apprentice.run(task_name, input_data)
            result = {
                "task_name": response.task_name,
                "output": response.output,
                "source": response.source.value,
                "status": response.status.value,
                "request_id": response.request_id,
                "duration_ms": response.duration_ms,
                "metadata": {
                    "model_id": response.metadata.model_id,
                    "cost_usd": response.metadata.cost_usd,
                    "fallback_used": response.metadata.fallback_used,
                    **self._package_run_metadata(data, package),
                },
            }
            await self._send_response(writer, 200, result)
        except Exception as e:
            await self._send_response(writer, 500, {"error": str(e)})

    async def _handle_status(self, writer: asyncio.StreamWriter) -> None:
        try:
            # Get status for all configured tasks
            tasks = []
            if hasattr(self._apprentice, '_task_registry') and hasattr(self._apprentice._task_registry, 'task_names'):
                for name in sorted(self._apprentice._task_registry.task_names):
                    try:
                        snap = await self._apprentice.status(name)
                        tasks.append({
                            "task_name": snap.task_name,
                            "confidence": snap.confidence_score,
                            "phase": snap.phase.value,
                            "budget_remaining": snap.budget_remaining_usd,
                        })
                    except Exception:
                        tasks.append({"task_name": name, "error": "unavailable"})

            package_state = getattr(self._apprentice, "_package_runtime_state", None)
            if package_state is not None and hasattr(package_state, "all_statuses"):
                known = {task.get("task_name") for task in tasks}
                for status in package_state.all_statuses():
                    if status.task_name in known:
                        continue
                    tasks.append(status.model_dump())

            await self._send_response(writer, 200, {"tasks": tasks, "timestamp": datetime.now(timezone.utc).isoformat()})
        except Exception as e:
            await self._send_response(writer, 500, {"error": str(e)})

    async def _handle_events(self, writer: asyncio.StreamWriter, body: str) -> None:
        """POST /v1/events — Fire-and-forget event ingestion. Always returns 200."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_response(writer, 400, {"error": f"Invalid JSON: {e}"})
            return

        try:
            package = getattr(self._apprentice, "_skill_package", None)
            if package is not None:
                result = await self._handle_packaged_event(data, package)
                await self._send_response(writer, 200, result)
            else:
                from apprentice.wos_models import WOSEvent
                from apprentice.wos_handler import handle_event
                event = WOSEvent(**data)
                result = await handle_event(self._apprentice, event)
                await self._send_response(writer, 200, result.model_dump())
        except Exception as e:
            # Fire-and-forget: always 200 even on internal errors
            print(f"[events] Event handler error (non-fatal): {e}", file=sys.stderr)
            await self._send_response(writer, 200, {"status": "accepted"})

    async def _handle_feedback(self, writer: asyncio.StreamWriter, body: str) -> None:
        """POST /v1/feedback — Record agent feedback."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_response(writer, 400, {"error": f"Invalid JSON: {e}"})
            return

        try:
            from apprentice.wos_models import FeedbackRecord
            from apprentice.wos_handler import handle_feedback
            feedback = FeedbackRecord(**data)
            result = await handle_feedback(self._apprentice, feedback)
            self._record_package_feedback(
                data.get("skill", ""),
                self._feedback_score(str(data.get("feedback_type", "edit"))),
            )
            await self._send_response(writer, 200, result.model_dump())
        except Exception as e:
            await self._send_response(writer, 500, {"error": str(e)})

    async def _handle_labeled_example(self, writer: asyncio.StreamWriter, body: str) -> None:
        """POST /v1/labeled-examples — Store externally generated draft/final pairs."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_response(writer, 400, {"error": f"Invalid JSON: {e}"})
            return

        task_name = data.get("task_name") or data.get("skill") or ""
        if not task_name:
            await self._send_response(writer, 400, {"error": "skill is required"})
            return

        package = getattr(self._apprentice, "_skill_package", None)
        if package is not None:
            check = self._package_constraint_check(data, package, scope="training_data")
            if not check["allowed"]:
                await self._send_response(writer, 422, {
                    "status": "blocked",
                    "package_id": package.package_id,
                    "skill": task_name,
                    "violations": check["violations"],
                })
                return

        input_data = data.get("input", {})
        draft_output = data.get("draft_output", data.get("agent_draft", {}))
        final_output = data.get("final_output", data.get("operator_final", data.get("edited_output", {})))
        if final_output in (None, {}):
            await self._send_response(writer, 400, {"error": "final_output is required"})
            return

        try:
            from apprentice.training_data_store import TrainingExample

            timestamp = data.get("timestamp") or datetime.now(timezone.utc).isoformat()
            metadata = {
                "source": "external_labeled_example",
                "request_id": data.get("request_id", ""),
                "feedback_type": data.get("feedback_type", "edit"),
                "action": data.get("action", ""),
                "method": data.get("method", ""),
                "draft_output": draft_output,
                **(data.get("metadata") if isinstance(data.get("metadata"), dict) else {}),
            }

            tds = getattr(self._apprentice, "_training_data_store", None)
            if tds is not None and hasattr(tds, "add_example"):
                example = TrainingExample(
                    input=json.dumps(input_data, sort_keys=True),
                    output=json.dumps(final_output, sort_keys=True),
                    task_type=task_name,
                    timestamp=timestamp,
                    metadata=metadata,
                )
                tds.add_example(example)

            ce = getattr(self._apprentice, "_confidence_engine", None)
            match_score = self._feedback_score(str(data.get("feedback_type", "edit")))
            self._record_package_feedback(task_name, match_score)
            if ce is not None and hasattr(ce, "record_comparison"):
                try:
                    ce.record_comparison(
                        task_id=task_name,
                        input_data=json.dumps(input_data, sort_keys=True),
                        local_output=json.dumps(draft_output, sort_keys=True),
                        remote_output=json.dumps(final_output, sort_keys=True),
                    )
                except Exception as e:
                    print(f"[labeled-examples] confidence update failed: {e}", file=sys.stderr)

            await self._send_response(writer, 200, {
                "status": "recorded",
                "skill": task_name,
                "match_score": match_score,
                "package_id": package.package_id if package is not None else None,
            })
        except Exception as e:
            await self._send_response(writer, 500, {"error": str(e)})

    async def _handle_constraint_check(self, writer: asyncio.StreamWriter, body: str) -> None:
        """POST /v1/constraints/check — Preflight package constraints without running a model."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_response(writer, 400, {"error": f"Invalid JSON: {e}"})
            return

        package = getattr(self._apprentice, "_skill_package", None)
        if package is None:
            await self._send_response(writer, 404, {"error": "No skill package registered"})
            return

        check = self._package_constraint_check(data, package)
        await self._send_response(writer, 200, {
            "package_id": package.package_id,
            "skill": data.get("task_name") or data.get("skill") or "",
            **check,
        })

    async def _handle_recommendations(self, writer: asyncio.StreamWriter, body: str) -> None:
        """POST /v1/recommendations — Get a recommendation from Apprentice."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_response(writer, 400, {"error": f"Invalid JSON: {e}"})
            return

        try:
            from apprentice.wos_models import RecommendRequest
            from apprentice.wos_handler import handle_recommend
            request = RecommendRequest(**data)
            result = await handle_recommend(self._apprentice, request)
            await self._send_response(writer, 200, result.model_dump())
        except Exception as e:
            await self._send_response(writer, 503, {"error": str(e)})

    async def _handle_skills(self, writer: asyncio.StreamWriter) -> None:
        """GET /v1/skills — List configured skills with phase/confidence."""
        try:
            package = getattr(self._apprentice, "_skill_package", None)
            if package is not None:
                await self._send_response(writer, 200, {
                    "skills": [
                        {
                            "name": skill.name,
                            "description": skill.description,
                            "methods": [method.model_dump() for method in skill.methods],
                            "actions": [action.model_dump() for action in skill.actions],
                            "outcomes": [outcome.model_dump() for outcome in skill.outcomes],
                            "constraints": [constraint.model_dump() for constraint in skill.constraints],
                            "tools": [tool.model_dump() for tool in skill.tools],
                            "evaluators": [evaluator.model_dump() for evaluator in skill.evaluators],
                            "artifacts": [artifact.model_dump() for artifact in skill.artifacts],
                        }
                        for skill in package.skills
                    ],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return

            from apprentice.wos_handler import list_skills
            skills = await list_skills(self._apprentice)
            await self._send_response(writer, 200, {
                "skills": [s.model_dump() for s in skills],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            await self._send_response(writer, 500, {"error": str(e)})

    async def _handle_skill_package(self, writer: asyncio.StreamWriter) -> None:
        """GET /v1/skill-package — Return registered external package metadata."""
        package = getattr(self._apprentice, "_skill_package", None)
        if package is None:
            await self._send_response(writer, 404, {"error": "No skill package registered"})
            return
        await self._send_response(writer, 200, package.model_dump())

    async def _handle_package_registry(self, writer: asyncio.StreamWriter) -> None:
        """GET /v1/package-registry — Return compact runtime registry metadata."""
        package = getattr(self._apprentice, "_skill_package", None)
        if package is None:
            await self._send_response(writer, 404, {"error": "No skill package registered"})
            return
        from apprentice.package_runtime import package_fingerprint, validate_package_runtime

        diagnostics = validate_package_runtime(package)
        registry_store = getattr(self._apprentice, "_package_registry_store", None)
        persisted = registry_store.load() if registry_store is not None else None
        await self._send_response(writer, 200, {
            "package_id": package.package_id,
            "schema_version": package.schema_version,
            "version": package.version,
            "fingerprint": package_fingerprint(package),
            "valid": diagnostics.ok,
            "diagnostics": [d.model_dump() for d in diagnostics.diagnostics],
            "persisted": persisted,
            "compatibility": package.compatibility.model_dump(),
            "runtime": package.runtime.model_dump(),
            "skills": [
                {
                    "name": skill.name,
                    "methods": [method.name for method in skill.methods],
                    "actions": [action.name for action in skill.actions],
                    "tools": [tool.name for tool in skill.tools if tool.enabled],
                    "evaluators": [evaluator.name for evaluator in skill.evaluators],
                    "artifacts": [artifact.name for artifact in skill.artifacts],
                }
                for skill in package.skills
            ],
            "event_mappings": [mapping.event_type for mapping in package.event_mappings],
        })

    async def _handle_tools(self, writer: asyncio.StreamWriter) -> None:
        """GET /v1/tools — List resolved tools from the loaded package."""
        package = getattr(self._apprentice, "_skill_package", None)
        if package is None:
            await self._send_response(writer, 404, {"error": "No skill package registered"})
            return
        await self._send_response(writer, 200, {
            "tools": [
                {
                    "skill": skill.name,
                    **tool.model_dump(),
                }
                for skill in package.skills
                for tool in package.resolved_tools(skill.name)
            ]
        })

    async def _handle_tool_call(self, writer: asyncio.StreamWriter, body: str) -> None:
        """POST /v1/tools/call — Invoke a supported package tool."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_response(writer, 400, {"error": f"Invalid JSON: {e}"})
            return

        package = getattr(self._apprentice, "_skill_package", None)
        if package is None:
            await self._send_response(writer, 404, {"error": "No skill package registered"})
            return

        skill_name = data.get("skill") or data.get("task_name") or ""
        tool_name = data.get("tool") or ""
        input_data = data.get("input", {})
        if not skill_name or not tool_name:
            await self._send_response(writer, 400, {"error": "skill and tool are required"})
            return

        check_data = {
            "skill": skill_name,
            "input": input_data,
            "metadata": data.get("metadata", {}),
            "action": data.get("action", ""),
            "method": data.get("method", ""),
        }
        check = self._package_constraint_check(check_data, package, scope="tool_call")
        hard_tool_violations = [
            violation for violation in check["violations"]
            if violation["kind"] == "hard"
        ]
        if hard_tool_violations:
            await self._send_response(writer, 422, {"status": "blocked", "violations": hard_tool_violations})
            return

        tool = next((t for t in package.resolved_tools(skill_name) if t.name == tool_name), None)
        if tool is None:
            await self._send_response(writer, 404, {"error": f"Tool not found: {tool_name}"})
            return

        from apprentice.package_runtime import invoke_tool

        result = await invoke_tool(
            tool,
            input_data,
            preflight_client=getattr(self._apprentice, "_tool_preflight_client", None),
        )
        await self._send_response(writer, 200, result.model_dump())

    async def _handle_evaluator_run(self, writer: asyncio.StreamWriter, body: str) -> None:
        """POST /v1/evaluators/run — Run a supported package evaluator."""
        try:
            data = json.loads(body)
        except json.JSONDecodeError as e:
            await self._send_response(writer, 400, {"error": f"Invalid JSON: {e}"})
            return

        package = getattr(self._apprentice, "_skill_package", None)
        if package is None:
            await self._send_response(writer, 404, {"error": "No skill package registered"})
            return

        skill_name = data.get("skill") or data.get("task_name") or ""
        evaluator_name = data.get("evaluator") or ""
        if not skill_name or not evaluator_name:
            await self._send_response(writer, 400, {"error": "skill and evaluator are required"})
            return
        skill = package.skill_for(skill_name)
        if skill is None:
            await self._send_response(writer, 404, {"error": f"Skill not found: {skill_name}"})
            return
        evaluator = next((e for e in skill.evaluators if e.name == evaluator_name), None)
        if evaluator is None:
            await self._send_response(writer, 404, {"error": f"Evaluator not found: {evaluator_name}"})
            return

        from apprentice.package_runtime import run_evaluator

        result = run_evaluator(evaluator, data.get("candidate"), data.get("reference"))
        await self._send_response(writer, 200, result.model_dump())

    def _package_constraint_check(self, data: dict[str, Any], package: Any, *, scope: str = "") -> dict[str, Any]:
        """Validate package method/action references and evaluate safe constraints."""
        task_name = data.get("task_name") or data.get("skill") or ""
        method_name = data.get("method")
        action_name = data.get("action")
        skill = package.skill_for(task_name)
        if skill is None:
            result = package.check_constraints(task_name, action_name=action_name, context={})
            return result.model_dump()

        if method_name and skill.method_for(method_name) is None:
            return {
                "allowed": False,
                "violations": [{
                    "name": "unknown_method",
                    "kind": "hard",
                    "expression": str(method_name),
                    "reason": f"method '{method_name}' is not defined for skill '{skill.name}'",
                }],
            }

        context = {
            "input": data.get("input", {}),
            "metadata": data.get("metadata", {}),
            "method": method_name or "",
            "action": action_name or "",
            "scope": scope,
            "package": {"id": package.package_id, "version": package.version},
            "pii": {"enabled": bool(getattr(self._apprentice, "_middleware_pipeline", None))},
        }
        result = package.check_constraints(task_name, action_name=action_name, context=context)
        return result.model_dump()

    def _package_run_metadata(self, data: dict[str, Any], package: Any) -> dict[str, Any]:
        if package is None:
            return {}
        return {
            "package_id": package.package_id,
            "method": data.get("method", ""),
            "action": data.get("action", ""),
        }

    def _feedback_score(self, feedback_type: str) -> float:
        scores = {
            "accept": 1.0,
            "edit": 0.5,
            "ai_score": 0.75,
            "ignore": 0.25,
            "reject": 0.0,
        }
        return scores.get(feedback_type, 0.5)

    def _record_package_feedback(self, task_name: str, score: float) -> None:
        package = getattr(self._apprentice, "_skill_package", None)
        state = getattr(self._apprentice, "_package_runtime_state", None)
        if package is None or state is None or not task_name:
            return
        if package.skill_for(task_name) is None:
            return
        if hasattr(state, "record_feedback"):
            state.record_feedback(task_name, score)

    async def _handle_packaged_event(self, data: dict[str, Any], package: Any) -> dict[str, Any]:
        """Map a host event through the registered skill package."""
        from apprentice.skill_package import read_path
        from apprentice.training_data_store import TrainingExample

        event_type = str(data.get("event_type", ""))
        mapping = package.mapping_for_event(event_type)
        if mapping is None:
            return {"status": "accepted", "mapped": False}

        payload = data.get("payload", data)
        input_data = read_path(data, mapping.input_path)
        output_data = read_path(data, mapping.output_path) if mapping.output_path else None
        feedback_data = read_path(data, mapping.feedback_path) if mapping.feedback_path else None
        subject_id = read_path(data, mapping.subject_path) if mapping.subject_path else data.get("subject_id", "")

        tds = getattr(self._apprentice, "_training_data_store", None)
        if tds is not None and hasattr(tds, "add_example"):
            example = TrainingExample(
                input=json.dumps(input_data if input_data is not None else payload, sort_keys=True),
                output=json.dumps(output_data if output_data is not None else feedback_data or {}, sort_keys=True),
                task_type=mapping.skill,
                timestamp=data.get("timestamp") or datetime.now(timezone.utc).isoformat(),
                metadata={
                    "source": "skill_package_event",
                    "package_id": package.package_id,
                    "event_type": event_type,
                    "subject_id": str(subject_id or ""),
                },
            )
            tds.add_example(example)

        return {
            "status": "accepted",
            "mapped": True,
            "package_id": package.package_id,
            "skill": mapping.skill,
        }

    async def _handle_report(self, writer: asyncio.StreamWriter) -> None:
        try:
            report = self._apprentice.report()
            result = {
                "total_runs": report.total_runs,
                "total_local_runs": report.total_local_runs,
                "total_remote_runs": report.total_remote_runs,
                "total_fallbacks": report.total_fallbacks,
                "total_errors": report.total_errors,
                "uptime_seconds": report.uptime_seconds,
                "timestamp": report.timestamp_utc,
            }
            await self._send_response(writer, 200, result)
        except Exception as e:
            await self._send_response(writer, 500, {"error": str(e)})

    async def _send_response(self, writer: asyncio.StreamWriter, status: int, body: dict) -> None:
        """Send an HTTP response."""
        status_text = {
            200: "OK", 400: "Bad Request", 401: "Unauthorized",
            403: "Forbidden", 404: "Not Found", 408: "Timeout",
            422: "Unprocessable Entity", 500: "Internal Server Error",
            503: "Service Unavailable",
        }
        body_bytes = json.dumps(body).encode("utf-8")
        response = (
            f"HTTP/1.1 {status} {status_text.get(status, 'Error')}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body_bytes)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode("utf-8") + body_bytes

        writer.write(response)
        await writer.drain()

    # ── Background Pipeline ───────────────────────────────────────────────

    async def _pipeline_loop(self) -> None:
        """Background loop that triggers training pipeline on interval."""
        while True:
            try:
                await asyncio.sleep(self._pipeline_interval)
                await self._run_pipeline()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[pipeline] Error: {e}", file=sys.stderr)

    async def _run_pipeline(self) -> None:
        """Single pipeline iteration: check training data, trigger fine-tuning, validate, promote."""
        apprentice = self._apprentice

        # Only run if factory-wired (has _ft_orchestrator)
        if not hasattr(apprentice, '_ft_orchestrator'):
            return

        if not hasattr(apprentice, '_real_config'):
            return

        # Guard: skip if a training job is already in flight
        if self._training_in_flight:
            print("[pipeline] Skipping: training job already in flight", file=sys.stderr)
            return

        cfg = apprentice._real_config
        tds = apprentice._training_data_store
        orchestrator = apprentice._ft_orchestrator

        for task_cfg in cfg.tasks:
            task_name = task_cfg.task_name
            try:
                count = tds.get_example_count(task_name)
                batch_size = cfg.finetuning.batch_size

                if count.training >= batch_size:
                    print(f"[pipeline] {task_name}: {count.training} examples >= {batch_size}, triggering fine-tune")

                    # Gather training examples
                    from apprentice.fine_tuning_orchestrator import TrainingExample as FTExample
                    import uuid

                    examples = tds.iter_training_examples(task_name)
                    ft_examples = []
                    for ex in examples[:batch_size]:
                        ft_examples.append(FTExample(
                            example_id=str(uuid.uuid4()),
                            user_prompt=ex.input,
                            assistant_response=ex.output,
                            collected_at=ex.timestamp,
                            source="training_data_store",
                        ))

                    if not ft_examples:
                        continue

                    # Orchestrator handles: run_id generation, batch validation,
                    # backend dispatch, failure tracking, and version persistence.
                    self._training_in_flight = True
                    try:
                        result = await asyncio.to_thread(
                            orchestrator.trigger_fine_tuning,
                            ft_examples,
                            cfg.finetuning.model_base,
                        )
                    finally:
                        self._training_in_flight = False

                    if not hasattr(result, 'status') or result.status != "success":
                        print(f"[pipeline] {task_name}: Fine-tune failed: {result}")
                        continue

                    print(f"[pipeline] {task_name}: Fine-tune succeeded: {result.version_id}")

                    # Validate and promote
                    if hasattr(apprentice, '_model_validator'):
                        try:
                            candidate_id = result.ollama_model_name or result.version_id
                            record = apprentice._model_validator.validate_and_promote(candidate_id)
                            if record.final_decision.value == "promoted":
                                print(f"[pipeline] {task_name}: Model promoted: {candidate_id}")
                            else:
                                print(f"[pipeline] {task_name}: Model not promoted: {record.final_decision.value}")
                        except Exception as e:
                            print(f"[pipeline] {task_name}: Validation error: {e}", file=sys.stderr)

            except Exception as e:
                self._training_in_flight = False
                print(f"[pipeline] {task_name}: Error: {e}", file=sys.stderr)


# ── Entry Point ───────────────────────────────────────────────────────────


async def serve_main(
    config_path: str,
    host: str = "127.0.0.1",
    port: int = 8710,
    pipeline_interval: int = 300,
    auth_mode: str = "none",
    api_key: str = "",
    jwt_secret: str = "",
    hmac_secret: str = "",
    tls_cert: str = "",
    tls_key: str = "",
    allowed_ips: Optional[list[str]] = None,
) -> None:
    """Main entry point for the serve daemon."""
    from apprentice.apprentice_class import Apprentice

    # Resolve auth secrets from env vars
    if api_key.startswith("env:"):
        api_key = os.environ.get(api_key[4:], "")
    if jwt_secret.startswith("env:"):
        jwt_secret = os.environ.get(jwt_secret[4:], "")
    if hmac_secret.startswith("env:"):
        hmac_secret = os.environ.get(hmac_secret[4:], "")

    security = SecurityConfig(
        auth_mode=auth_mode,
        api_key=api_key,
        jwt_secret=jwt_secret,
        hmac_secret=hmac_secret,
        tls_cert=tls_cert,
        tls_key=tls_key,
        allowed_ips=allowed_ips or [],
    )

    print(f"Loading config from {config_path}...")
    apprentice = await Apprentice.from_config(config_path)

    scheme = "https" if tls_cert else "http"

    async with apprentice:
        server = ApprenticeServer(apprentice, host, port, pipeline_interval, security)
        await server.start()
        print(f"Apprentice serving on {host}:{port}")
        print(f"  Health:          {scheme}://{host}:{port}/health")
        print(f"  Run:             POST {scheme}://{host}:{port}/v1/run")
        print(f"  Status:          {scheme}://{host}:{port}/v1/status")
        print(f"  Report:          {scheme}://{host}:{port}/v1/report")
        print(f"  Events:          POST {scheme}://{host}:{port}/v1/events")
        print(f"  Feedback:        POST {scheme}://{host}:{port}/v1/feedback")
        print(f"  Recommend:       POST {scheme}://{host}:{port}/v1/recommendations")
        print(f"  Skills:          GET  {scheme}://{host}:{port}/v1/skills")
        print(f"  Skill package:   GET  {scheme}://{host}:{port}/v1/skill-package")
        print(f"  Pipeline interval: {pipeline_interval}s")
        print(f"  Security: auth={auth_mode}", end="")
        if tls_cert:
            print(f", tls=on", end="")
        if allowed_ips:
            print(f", allowed_ips={allowed_ips}", end="")
        print()
        await server.wait_for_shutdown()
