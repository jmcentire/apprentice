"""
CLI entry point for Apprentice.

Thin shell with three responsibilities:
1. Parse arguments via argparse
2. Load config and construct Apprentice instance
3. Format and print results
"""

import argparse
import asyncio
import inspect
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

try:
    from .cli_models import (
        ApprenticeConfig,
        BudgetInfo,
        GlobalFlags,
        IngestArgs,
        InitArgs,
        InputData,
        ParsedArgs,
        PhaseInfo,
        PIIEvaluateArgs,
        PIIIngestArgs,
        ReportArgs,
        ReportResult,
        RunArgs,
        RunResult,
        ServeArgs,
        StatusArgs,
        StatusResult,
        SubcommandName,
        TaskStatusEntry,
        CLIResult,
    )
except ImportError:
    from cli_models import (
        ApprenticeConfig,
        BudgetInfo,
        GlobalFlags,
        IngestArgs,
        InitArgs,
        InputData,
        ParsedArgs,
        PhaseInfo,
        PIIEvaluateArgs,
        PIIIngestArgs,
        ReportArgs,
        ReportResult,
        RunArgs,
        RunResult,
        ServeArgs,
        StatusArgs,
        StatusResult,
        SubcommandName,
        TaskStatusEntry,
        CLIResult,
    )


class JSONFormatter(logging.Formatter):
    """Structured JSON formatter for logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        return json.dumps(log_obj)


def configure_logging(verbose: int) -> None:
    """
    Configure structured JSON logging to stderr.

    verbose=0 -> WARNING
    verbose=1 -> INFO
    verbose=2 -> DEBUG
    """
    # Map verbose count to log level
    level_map = {
        0: logging.WARNING,
        1: logging.INFO,
        2: logging.DEBUG,
    }
    level = level_map.get(verbose, logging.DEBUG)

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers
    root.handlers.clear()

    # Add stderr handler with JSON formatter
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)


def parse_args(argv: list[str]) -> ParsedArgs:
    """
    Parse command-line arguments into structured ParsedArgs model.

    Raises SystemExit on invalid arguments.
    """
    parser = argparse.ArgumentParser(
        prog="apprentice",
        description="Apprentice CLI - Adaptive response threshold system"
    )

    # Global flags
    parser.add_argument(
        "--config",
        dest="config_path",
        default="./apprentice.yaml",
        help="Path to config file (default: ./apprentice.yaml)"
    )
    parser.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        help="Output JSON instead of human-readable format"
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Execute a task")
    run_parser.add_argument("task_name", help="Name of task to execute")
    run_parser.add_argument("--input", dest="input_raw", required=True, help="Input JSON or @file")

    # status subcommand
    status_parser = subparsers.add_parser("status", help="Show task status")
    status_parser.add_argument("--task", dest="task_filter", default=None, help="Filter by task name")

    # report subcommand
    report_parser = subparsers.add_parser("report", help="Generate full report")
    report_parser.add_argument("--output", dest="output_path", default=None, help="Write to file")

    # init subcommand
    init_parser = subparsers.add_parser("init", help="Interactive setup wizard")
    init_parser.add_argument("--output", dest="init_output", default="./apprentice.yaml", help="Output config path")

    # serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start HTTP daemon")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8710, help="Bind port (default: 8710)")
    serve_parser.add_argument("--pipeline-interval", type=int, default=300, help="Pipeline interval seconds (default: 300)")
    serve_parser.add_argument("--auth", dest="auth_mode", default="none",
                              choices=["none", "api-key", "jwt", "hmac"],
                              help="Auth mode (default: none)")
    serve_parser.add_argument("--api-key", dest="serve_api_key", default="",
                              help="API key or env:VAR_NAME for api-key auth")
    serve_parser.add_argument("--jwt-secret", default="",
                              help="JWT secret or env:VAR_NAME for jwt auth")
    serve_parser.add_argument("--hmac-secret", default="",
                              help="HMAC secret or env:VAR_NAME for hmac auth")
    serve_parser.add_argument("--tls-cert", default="", help="Path to TLS certificate")
    serve_parser.add_argument("--tls-key", default="", help="Path to TLS private key")
    serve_parser.add_argument("--allowed-ips", default="",
                              help="Comma-separated IP allowlist (CIDRs or addresses)")

    # pii-ingest subcommand
    pii_ingest_parser = subparsers.add_parser("pii-ingest", help="Ingest PII training data")
    pii_ingest_parser.add_argument("--dataset", default="ai4privacy/pii-masking-200k",
                                    help="HuggingFace dataset ID (default: ai4privacy/pii-masking-200k)")
    pii_ingest_parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    pii_ingest_parser.add_argument("--limit", type=int, default=10000, help="Max examples to ingest (default: 10000)")

    # pii-evaluate subcommand
    pii_evaluate_parser = subparsers.add_parser("pii-evaluate", help="Evaluate PII detection quality")
    pii_evaluate_parser.add_argument("--mode", default="regex_only",
                                      choices=["regex_only", "hybrid", "ner_only"],
                                      help="Detection mode (default: regex_only)")
    pii_evaluate_parser.add_argument("--limit", type=int, default=1000, help="Max examples to evaluate (default: 1000)")

    # ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Bulk-load training examples from file")
    ingest_parser.add_argument("file", help="Path to JSONL or CSV file")
    ingest_parser.add_argument("--format", dest="ingest_format", default="auto",
                               choices=["auto", "jsonl", "csv"],
                               help="File format (default: auto-detect from extension)")
    ingest_parser.add_argument("--task", dest="ingest_task", default=None,
                               help="Override task_type for all rows (required for CSV without task_type column)")

    # Parse
    args = parser.parse_args(argv)

    # Clamp verbose to [0, 2]
    verbose = min(args.verbose, 2)

    # Build structured model
    global_flags = GlobalFlags(
        config_path=args.config_path,
        json_mode=args.json_mode,
        verbose=verbose,
    )

    command = SubcommandName(args.command)

    # Populate subcommand-specific args
    run_args = None
    status_args = None
    report_args = None
    init_args = None
    serve_args = None
    ingest_args = None
    pii_ingest_args = None
    pii_evaluate_args = None

    if command == SubcommandName.run:
        run_args = RunArgs(
            task_name=args.task_name,
            input_raw=args.input_raw,
        )
    elif command == SubcommandName.status:
        status_args = StatusArgs(task_filter=args.task_filter)
    elif command == SubcommandName.report:
        report_args = ReportArgs(output_path=args.output_path)
    elif command == SubcommandName.init:
        init_args = InitArgs(output_path=args.init_output)
    elif command == SubcommandName.serve:
        serve_args = ServeArgs(
            host=args.host,
            port=args.port,
            pipeline_interval=args.pipeline_interval,
            auth_mode=args.auth_mode,
            api_key=args.serve_api_key,
            jwt_secret=args.jwt_secret,
            hmac_secret=args.hmac_secret,
            tls_cert=args.tls_cert,
            tls_key=args.tls_key,
            allowed_ips=args.allowed_ips,
        )
    elif command == SubcommandName.ingest:
        ingest_args = IngestArgs(
            file_path=args.file,
            format=args.ingest_format,
            task=args.ingest_task,
        )
    elif command == SubcommandName.pii_ingest:
        pii_ingest_args = PIIIngestArgs(
            dataset=args.dataset,
            split=args.split,
            limit=args.limit,
        )
    elif command == SubcommandName.pii_evaluate:
        pii_evaluate_args = PIIEvaluateArgs(
            mode=args.mode,
            limit=args.limit,
        )

    return ParsedArgs(
        global_flags=global_flags,
        command=command,
        run_args=run_args,
        status_args=status_args,
        report_args=report_args,
        init_args=init_args,
        serve_args=serve_args,
        ingest_args=ingest_args,
        pii_ingest_args=pii_ingest_args,
        pii_evaluate_args=pii_evaluate_args,
    )


def load_config(path: str) -> ApprenticeConfig:
    """
    Load and validate YAML config file.

    Raises:
        FileNotFoundError: Config file not found
        ValueError: Invalid YAML or validation error
    """
    if not path:
        raise ValueError("Config path cannot be empty")

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file: {path}") from e

    if not isinstance(data, dict):
        raise ValueError("Config must be a YAML object")

    # Basic validation - config should have required fields
    # This is a minimal check; real validation would be done by Apprentice
    if not data:
        raise ValueError("Config cannot be empty")

    # Check for expected config structure
    # A valid ApprenticeConfig should have certain required fields
    # For this test, we validate that it's not just random keys
    expected_fields = {'tasks', 'models', 'budget', 'local_model', 'remote_model'}
    if not any(field in data for field in expected_fields):
        raise ValueError(
            f"Config validation failed: expected at least one of {expected_fields}, "
            f"but got only {set(data.keys())}"
        )

    # Validate via pydantic (strict mode)
    try:
        return ApprenticeConfig(config_data=data)
    except Exception as e:
        raise ValueError(f"Config validation failed: {e}") from e


def resolve_input(input_raw: str) -> InputData:
    """
    Resolve input string to a dict.

    If input starts with '@', read from file.
    Otherwise, parse as JSON.

    Raises:
        ValueError: Invalid JSON or not a JSON object
        FileNotFoundError: Referenced file not found
    """
    if input_raw.startswith("@"):
        # File reference
        file_path = input_raw[1:]
        if not file_path:
            raise ValueError("Empty file path after @")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input file {file_path}: {e}") from e
    else:
        # Direct JSON string
        try:
            data = json.loads(input_raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}") from e

    # Validate that it's a JSON object (dict)
    if not isinstance(data, dict):
        raise ValueError(f"Input must be a JSON object, got {type(data).__name__}")

    return InputData(data=data)


def format_output(result: CLIResult, json_mode: bool) -> str:
    """
    Format a result for terminal output.

    Returns string without trailing newline.
    """
    if json_mode:
        # JSON mode - compact serialization
        return result.model_dump_json(exclude_none=False)
    else:
        # Human mode - formatted text
        if isinstance(result, RunResult):
            lines = [
                f"Task: {result.task_name}",
                f"Success: {result.success}",
            ]
            if result.error_message:
                lines.append(f"Error: {result.error_message}")
            else:
                lines.append(f"Output: {result.output}")
            return "\n".join(lines)

        elif isinstance(result, StatusResult):
            lines = [f"Status Report - {result.timestamp}", ""]
            for task in result.tasks:
                lines.append(f"Task: {task.task_name}")
                lines.append(f"  Phase: {task.phase.phase_name}")
                lines.append(f"  Confidence: {task.phase.confidence_score:.2f}")
                lines.append(f"  Local Primary: {task.phase.is_local_primary}")
                lines.append(f"  Budget: {task.budget.budget_spent:.2f} / {task.budget.budget_limit:.2f}")
                lines.append("")
            return "\n".join(lines).rstrip()

        elif isinstance(result, ReportResult):
            lines = [
                f"System Report - {result.timestamp}",
                f"Config: {result.config_path}",
                f"Uptime: {result.system_uptime_seconds:.1f}s",
                f"Budget: {result.total_budget_spent:.2f} / {result.total_budget_limit:.2f}",
                "",
                "Tasks:",
            ]
            for task in result.tasks:
                lines.append(f"  - {task.task_name}")
                lines.append(f"    Phase: {task.phase.phase_name}")
                lines.append(f"    Confidence: {task.phase.confidence_score:.2f}")
            if result.written_to_file:
                lines.append("")
                lines.append(f"Written to: {result.written_to_file}")
            return "\n".join(lines)

        else:
            # Fallback
            return str(result)


async def execute_run(apprentice: Any, run_args: RunArgs) -> RunResult:
    """
    Execute the 'run' subcommand.

    Resolves input and calls Apprentice.run().
    """
    # Resolve input
    input_data = resolve_input(run_args.input_raw)

    try:
        # Call apprentice
        response = await apprentice.run(run_args.task_name, input_data.data)

        # Wrap in RunResult
        return RunResult(
            task_name=run_args.task_name,
            output=response.output,
            success=response.success,
            error_message="",
        )
    except Exception as e:
        # Return error result
        return RunResult(
            task_name=run_args.task_name,
            output={},
            success=False,
            error_message=str(e),
        )


async def execute_status(apprentice: Any, status_args: StatusArgs) -> StatusResult:
    """
    Execute the 'status' subcommand.

    Calls Apprentice.status_all() and maps ConfidenceSnapshot to CLI models.
    """
    snapshots = await apprentice.status_all()

    tasks = []
    for snap in snapshots:
        phase = PhaseInfo(
            phase_name=snap.phase.value if hasattr(snap.phase, 'value') else str(snap.phase),
            confidence_score=snap.confidence_score,
            is_local_primary=snap.local_model_available,
        )
        budget = BudgetInfo(
            budget_limit=snap.budget_used_usd + snap.budget_remaining_usd,
            budget_spent=snap.budget_used_usd,
            budget_remaining=snap.budget_remaining_usd,
            is_exhausted=snap.budget_exhausted,
        )
        tasks.append(TaskStatusEntry(
            task_name=snap.task_name,
            phase=phase,
            budget=budget,
        ))

    if status_args.task_filter:
        tasks = [t for t in tasks if t.task_name == status_args.task_filter]

    return StatusResult(
        tasks=tasks,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def execute_report(
    apprentice: Any,
    report_args: ReportArgs,
    config_path: str,
    json_mode: bool,
) -> ReportResult:
    """
    Execute the 'report' subcommand.

    Calls Apprentice.report() and optionally writes to file.
    """
    # Get report data (report() may be sync or async depending on implementation)
    raw_report = apprentice.report()
    if inspect.isawaitable(raw_report):
        raw_report = await raw_report

    # Convert tasks
    tasks = []
    for entry in raw_report.tasks:
        phase = PhaseInfo(
            phase_name=entry.phase.phase_name,
            confidence_score=entry.phase.confidence_score,
            is_local_primary=entry.phase.is_local_primary,
        )
        budget = BudgetInfo(
            budget_limit=entry.budget.budget_limit,
            budget_spent=entry.budget.budget_spent,
            budget_remaining=entry.budget.budget_remaining,
            is_exhausted=entry.budget.is_exhausted,
        )
        task_entry = TaskStatusEntry(
            task_name=entry.task_name,
            phase=phase,
            budget=budget,
        )
        tasks.append(task_entry)

    result = ReportResult(
        tasks=tasks,
        total_budget_limit=raw_report.total_budget_limit,
        total_budget_spent=raw_report.total_budget_spent,
        system_uptime_seconds=raw_report.system_uptime_seconds,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_path=config_path,
        written_to_file="",
    )

    # Write to file if requested
    if report_args.output_path:
        output = format_output(result, json_mode)
        try:
            with open(report_args.output_path, "w") as f:
                f.write(output)
            result.written_to_file = report_args.output_path
        except OSError as e:
            raise OSError(f"Failed to write report to {report_args.output_path}: {e}") from e

    return result


def execute_pii_ingest(pii_ingest_args: PIIIngestArgs) -> int:
    """Execute the 'pii-ingest' subcommand.

    Downloads and ingests a PII detection dataset from HuggingFace.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library required. Install with: pip install datasets", file=sys.stderr)
        return 1

    print(f"Loading dataset '{pii_ingest_args.dataset}' (split={pii_ingest_args.split}, limit={pii_ingest_args.limit})...")

    try:
        ds = load_dataset(
            pii_ingest_args.dataset,
            split=f"{pii_ingest_args.split}[:{pii_ingest_args.limit}]",
        )
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        return 1

    # Store examples as JSON-lines
    output_dir = Path(".apprentice/pii_training_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pii_detection_examples.jsonl"

    count = 0
    with open(output_file, "w", encoding="utf-8") as fh:
        for row in ds:
            example = {
                "task_type": "pii_detection",
                "source_text": row.get("source_text", row.get("text", "")),
                "privacy_mask": row.get("privacy_mask", {}),
                "span_labels": row.get("span_labels", []),
                "bio_labels": row.get("bio_labels", []),
                "tokenised_text": row.get("tokenised_text", ""),
            }
            fh.write(json.dumps(example) + "\n")
            count += 1

    print(f"Ingested {count} examples to {output_file}")
    return 0


def execute_pii_evaluate(pii_evaluate_args: PIIEvaluateArgs) -> int:
    """Execute the 'pii-evaluate' subcommand.

    Runs detection strategies against labeled data and prints F1 metrics.
    """
    from apprentice.pii_detection import (
        CompositeDetectionPipeline,
        FieldHeuristicStrategy,
        NERDetectionStrategy,
        PIIDetection,
        RegexDetectionStrategy,
    )
    from apprentice.pii_evaluation import PIIDetectionEvaluator

    # Load examples
    examples_file = Path(".apprentice/pii_training_data/pii_detection_examples.jsonl")
    if not examples_file.exists():
        print("Error: No training data found. Run 'apprentice pii-ingest' first.", file=sys.stderr)
        return 1

    examples = []
    with open(examples_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not examples:
        print("Error: No valid examples found.", file=sys.stderr)
        return 1

    # Limit examples
    examples = examples[:pii_evaluate_args.limit]

    # Build detection pipeline based on mode
    pipeline = CompositeDetectionPipeline()
    mode = pii_evaluate_args.mode

    if mode in ("regex_only", "hybrid"):
        pipeline.add_strategy(RegexDetectionStrategy())
        pipeline.add_strategy(FieldHeuristicStrategy())

    if mode in ("hybrid", "ner_only"):
        pipeline.add_strategy(NERDetectionStrategy())

    evaluator = PIIDetectionEvaluator()

    # Run evaluation
    all_predictions: list[PIIDetection] = []
    all_ground_truth: list[PIIDetection] = []
    offset = 0

    print(f"Evaluating {len(examples)} examples with mode={mode}...")

    for ex in examples:
        text = ex.get("source_text", "")
        if not text:
            continue

        # Get predictions
        preds = pipeline.detect_all(text)
        # Offset predictions for global evaluation
        for p in preds:
            all_predictions.append(PIIDetection(
                start=p.start + offset,
                end=p.end + offset,
                value=p.value,
                entity_type=p.entity_type,
                confidence=p.confidence,
                source=p.source,
            ))

        # Parse ground truth from span_labels if available
        span_labels = ex.get("span_labels", [])
        if isinstance(span_labels, list):
            for sl in span_labels:
                if isinstance(sl, dict):
                    gt_start = sl.get("start", 0)
                    gt_end = sl.get("end", 0)
                    gt_type = sl.get("label", sl.get("type", "unknown")).lower()
                    gt_value = text[gt_start:gt_end] if gt_start < gt_end <= len(text) else ""
                    all_ground_truth.append(PIIDetection(
                        start=gt_start + offset,
                        end=gt_end + offset,
                        value=gt_value,
                        entity_type=gt_type,
                        confidence=1.0,
                        source="ground_truth",
                    ))

        offset += len(text) + 1

    result = evaluator.evaluate(all_predictions, all_ground_truth)

    # Print results
    print(f"\n{'Entity Type':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 79)
    for m in result.entity_metrics:
        print(f"{m.entity_type:<25} {m.precision:>10.3f} {m.recall:>10.3f} {m.f1:>10.3f} {m.true_positives:>6} {m.false_positives:>6} {m.false_negatives:>6}")
    print("-" * 79)
    print(f"{'Micro Average':<25} {result.micro_precision:>10.3f} {result.micro_recall:>10.3f} {result.micro_f1:>10.3f}")
    print(f"{'Macro Average':<25} {result.macro_precision:>10.3f} {result.macro_recall:>10.3f} {result.macro_f1:>10.3f}")
    print(f"\nTotal predictions: {result.total_predictions}, Total ground truth: {result.total_ground_truth}")

    return 0


def execute_ingest(ingest_args: IngestArgs, config_path: str) -> dict:
    """
    Execute the 'ingest' subcommand.

    Reads a JSONL or CSV file and bulk-loads training examples into the store.
    Does not require a full Apprentice instance — only the training data store.
    """
    import csv as csv_mod
    from apprentice.training_data_store import (
        TrainingDataStore,
        TrainingDataStoreConfig,
        TrainingExample,
    )
    from apprentice import config_loader

    file_path = Path(ingest_args.file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Ingest file not found: {file_path}")

    # Detect format
    fmt = ingest_args.format
    if fmt == "auto":
        suffix = file_path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            fmt = "jsonl"
        elif suffix == ".csv":
            fmt = "csv"
        else:
            raise ValueError(f"Cannot auto-detect format for '{suffix}'. Use --format jsonl|csv")

    # Load config to get training data store path
    cfg = config_loader.load_config(Path(config_path))
    store = TrainingDataStore()
    store.initialize(TrainingDataStoreConfig(
        base_dir=cfg.training_data.storage_dir,
    ))

    # Parse file into TrainingExample objects
    examples: list[TrainingExample] = []
    now = datetime.now(timezone.utc).isoformat()

    if fmt == "jsonl":
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                task = ingest_args.task or row.get("task_type", "")
                if not task:
                    raise ValueError(f"Line {line_num}: missing task_type (use --task to set)")
                examples.append(TrainingExample(
                    input=row["input"],
                    output=row["output"],
                    task_type=task,
                    timestamp=row.get("timestamp", now),
                    metadata=row.get("metadata", {}),
                ))
    elif fmt == "csv":
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for line_num, row in enumerate(reader, 2):
                task = ingest_args.task or row.get("task_type", "")
                if not task:
                    raise ValueError(f"Row {line_num}: missing task_type (use --task to set)")
                examples.append(TrainingExample(
                    input=row["input"],
                    output=row["output"],
                    task_type=task,
                    timestamp=row.get("timestamp", now),
                    metadata=json.loads(row["metadata"]) if row.get("metadata") else {},
                ))

    if not examples:
        return {"status": "empty", "count": 0, "message": "No examples found in file"}

    results = store.add_examples(examples)
    total = sum(r.example_count for r in results)
    tasks_loaded = {r.task_type: r.example_count for r in results}

    return {
        "status": "ok",
        "count": len(examples),
        "tasks": tasks_loaded,
        "message": f"Loaded {len(examples)} examples across {len(tasks_loaded)} task(s)",
    }


def main(argv: Optional[list[str]] = None) -> int:
    """
    Main CLI entry point.

    Returns exit code:
        0 - success
        1 - application/config error
        2 - usage error
        130 - keyboard interrupt
    """
    # Handle SIGPIPE on Unix to prevent BrokenPipeError tracebacks
    if hasattr(signal, 'SIGPIPE'):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    if argv is None:
        argv = sys.argv[1:]

    try:
        # Parse arguments
        try:
            args = parse_args(argv)
        except SystemExit as e:
            # argparse raises SystemExit on error
            # Exit code 2 for usage errors
            return 2

        # Configure logging before any other work
        configure_logging(args.global_flags.verbose)

        # Handle init subcommand (no config needed)
        if args.command == SubcommandName.init:
            try:
                from .init_wizard import run_wizard
            except ImportError:
                from init_wizard import run_wizard
            output_path = args.init_args.output_path if args.init_args else "./apprentice.yaml"
            run_wizard(output_path)
            return 0

        # Handle serve subcommand
        if args.command == SubcommandName.serve:
            try:
                from .serve import serve_main
            except ImportError:
                from serve import serve_main
            serve = args.serve_args
            host = serve.host if serve else "127.0.0.1"
            port = serve.port if serve else 8710
            interval = serve.pipeline_interval if serve else 300
            allowed_ips = [ip.strip() for ip in serve.allowed_ips.split(",") if ip.strip()] if serve and serve.allowed_ips else []
            asyncio.run(serve_main(
                config_path=args.global_flags.config_path,
                host=host, port=port, pipeline_interval=interval,
                auth_mode=serve.auth_mode if serve else "none",
                api_key=serve.api_key if serve else "",
                jwt_secret=serve.jwt_secret if serve else "",
                hmac_secret=serve.hmac_secret if serve else "",
                tls_cert=serve.tls_cert if serve else "",
                tls_key=serve.tls_key if serve else "",
                allowed_ips=allowed_ips,
            ))
            return 0

        # Handle pii-ingest subcommand (no config needed)
        if args.command == SubcommandName.pii_ingest:
            return execute_pii_ingest(args.pii_ingest_args)

        # Handle pii-evaluate subcommand (no config needed)
        if args.command == SubcommandName.pii_evaluate:
            return execute_pii_evaluate(args.pii_evaluate_args)

        # Check for unresolved env vars and warn on stderr
        try:
            from apprentice import config_loader as _cl
            unresolved = _cl.get_unresolved_vars()
            if unresolved:
                for var_name, field_path in unresolved:
                    print(
                        f"\u26a0 {var_name} not set (referenced by {field_path})",
                        file=sys.stderr,
                    )
                print(
                    "\u2192 Commands requiring API access will fail. "
                    "Set these in your shell profile.",
                    file=sys.stderr,
                )
        except ImportError:
            pass

        # Handle ingest subcommand (needs config but not full Apprentice)
        if args.command == SubcommandName.ingest:
            try:
                result = execute_ingest(args.ingest_args, args.global_flags.config_path)
                if args.global_flags.json_mode:
                    output = json.dumps(result)
                else:
                    output = result.get("message", str(result))
                print(output, file=sys.stdout)
                return 0
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
                logging.error(f"Ingest error: {e}")
                print(f"Error: {e}", file=sys.stderr)
                return 1

        # Load config
        try:
            config = load_config(args.global_flags.config_path)
        except (FileNotFoundError, ValueError) as e:
            logging.error(f"Config error: {e}")
            print(f"Error: {e}", file=sys.stderr)
            return 1

        # Construct the real Apprentice from the config file
        try:
            from apprentice.apprentice_class import Apprentice
        except ImportError:
            logging.error("Apprentice package not properly installed")
            print("Error: apprentice package not properly installed", file=sys.stderr)
            return 1

        # For 'run' command, fail early if provider API key is unresolved
        if args.command == SubcommandName.run:
            try:
                from apprentice import config_loader as _cl2
                for var_name, field_path in _cl2.get_unresolved_vars():
                    if field_path == "provider.api_key":
                        print(
                            f"Error: Cannot run task — {var_name} is not set. "
                            f"Set it in your shell profile or environment.",
                            file=sys.stderr,
                        )
                        return 1
            except ImportError:
                pass

        # Execute the appropriate subcommand within Apprentice async context
        async def async_main():
            try:
                apprentice = await Apprentice.from_config(args.global_flags.config_path)
            except (FileNotFoundError, ValueError, TypeError) as e:
                logging.error(f"Failed to initialize Apprentice: {e}")
                print(f"Error: {e}", file=sys.stderr)
                raise

            async with apprentice:
                if args.command == SubcommandName.run:
                    return await execute_run(apprentice, args.run_args)
                elif args.command == SubcommandName.status:
                    return await execute_status(apprentice, args.status_args)
                elif args.command == SubcommandName.report:
                    return await execute_report(
                        apprentice,
                        args.report_args,
                        args.global_flags.config_path,
                        args.global_flags.json_mode
                    )

        result = asyncio.run(async_main())

        # Format and print output
        output = format_output(result, args.global_flags.json_mode)
        print(output, file=sys.stdout)

        return 0

    except KeyboardInterrupt:
        # User pressed Ctrl+C
        print("", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # SIGPIPE - exit silently
        return 0
    except Exception as e:
        # Any other error
        logging.error(f"Unexpected error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ── Auto-injected export aliases (Pact export gate) ──
TaskStatusEntryList = TaskStatusEntry
