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

        task_name = data.get("task_name", "")
        input_data = data.get("input", {})

        if not task_name:
            await self._send_response(writer, 400, {"error": "task_name is required"})
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

            await self._send_response(writer, 200, {"tasks": tasks, "timestamp": datetime.now(timezone.utc).isoformat()})
        except Exception as e:
            await self._send_response(writer, 500, {"error": str(e)})

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
            500: "Internal Server Error",
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

        cfg = apprentice._real_config
        tds = apprentice._training_data_store
        ft_backend = apprentice._ft_orchestrator

        for task_cfg in cfg.tasks:
            task_name = task_cfg.task_name
            try:
                count = tds.get_example_count(task_name)
                batch_size = cfg.finetuning.batch_size

                if count.training >= batch_size:
                    print(f"[pipeline] {task_name}: {count.training} examples >= {batch_size}, triggering fine-tune")

                    # Gather training examples
                    from apprentice.fine_tuning_orchestrator import FineTuneRequest, TrainingExample as FTExample
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

                    request = FineTuneRequest(
                        run_id=str(uuid.uuid4()),
                        base_model=cfg.finetuning.model_base,
                        training_examples=ft_examples,
                    )

                    result = ft_backend.fine_tune(request)

                    if not hasattr(result, 'status') or result.status != "success":
                        print(f"[pipeline] {task_name}: Fine-tune failed: {result}")
                        continue

                    print(f"[pipeline] {task_name}: Fine-tune succeeded: {result.version_id}")

                    # Save model version
                    if hasattr(apprentice, '_ft_version_store'):
                        try:
                            apprentice._ft_version_store.save_model_version(result)
                        except Exception as e:
                            print(f"[pipeline] {task_name}: Version store error: {e}", file=sys.stderr)

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
        print(f"  Health:  {scheme}://{host}:{port}/health")
        print(f"  Run:     POST {scheme}://{host}:{port}/v1/run")
        print(f"  Status:  {scheme}://{host}:{port}/v1/status")
        print(f"  Report:  {scheme}://{host}:{port}/v1/report")
        print(f"  Pipeline interval: {pipeline_interval}s")
        print(f"  Security: auth={auth_mode}", end="")
        if tls_cert:
            print(f", tls=on", end="")
        if allowed_ips:
            print(f", allowed_ips={allowed_ips}", end="")
        print()
        await server.wait_for_shutdown()
