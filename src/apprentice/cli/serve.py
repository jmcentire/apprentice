"""HTTP Daemon for Apprentice.

Bare asyncio HTTP server with background training pipeline loop.
No external framework dependencies.
"""

import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Optional


class ApprenticeServer:
    """Async HTTP server wrapping an Apprentice instance."""

    def __init__(self, apprentice: Any, host: str = "0.0.0.0", port: int = 8710, pipeline_interval: int = 300):
        self._apprentice = apprentice
        self._host = host
        self._port = port
        self._pipeline_interval = pipeline_interval
        self._server: Optional[asyncio.AbstractServer] = None
        self._pipeline_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the HTTP server and background pipeline."""
        self._server = await asyncio.start_server(
            self._handle_connection, self._host, self._port,
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
        """Handle a single HTTP connection."""
        try:
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

            # Extract body
            body = ""
            if "\r\n\r\n" in request_text:
                body = request_text.split("\r\n\r\n", 1)[1]

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
        status_text = {200: "OK", 400: "Bad Request", 404: "Not Found", 408: "Timeout", 500: "Internal Server Error"}
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
        """Single pipeline iteration: check training data, trigger fine-tuning."""
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

                    if hasattr(result, 'status') and result.status == "success":
                        print(f"[pipeline] {task_name}: Fine-tune succeeded: {result.version_id}")
                    else:
                        print(f"[pipeline] {task_name}: Fine-tune returned: {result}")

            except Exception as e:
                print(f"[pipeline] {task_name}: Error: {e}", file=sys.stderr)


async def serve_main(config_path: str, host: str = "0.0.0.0", port: int = 8710, pipeline_interval: int = 300) -> None:
    """Main entry point for the serve daemon."""
    from apprentice.apprentice_class import Apprentice

    print(f"Loading config from {config_path}...")
    apprentice = await Apprentice.from_config(config_path)

    async with apprentice:
        server = ApprenticeServer(apprentice, host, port, pipeline_interval)
        await server.start()
        print(f"Apprentice serving on {host}:{port}")
        print(f"  Health:  http://{host}:{port}/health")
        print(f"  Run:     POST http://{host}:{port}/v1/run")
        print(f"  Status:  http://{host}:{port}/v1/status")
        print(f"  Report:  http://{host}:{port}/v1/report")
        print(f"  Pipeline interval: {pipeline_interval}s")
        await server.wait_for_shutdown()
