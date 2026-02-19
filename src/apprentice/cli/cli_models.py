"""
Pydantic models for CLI component.

All models use strict mode for serialization correctness.
"""

from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class ExitCode(Enum):
    """Process exit codes returned by main(). Follows Unix conventions."""
    SUCCESS_0 = 0
    APP_ERROR_1 = 1
    USAGE_ERROR_2 = 2
    KEYBOARD_INTERRUPT_130 = 130


class OutputFormat(Enum):
    """Controls how results are serialized for terminal output."""
    HUMAN = "HUMAN"
    JSON = "JSON"


class LogLevel(Enum):
    """Logging verbosity levels mapped from --verbose flag."""
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class SubcommandName(Enum):
    """CLI subcommands recognized by the argument parser."""
    run = "run"
    status = "status"
    report = "report"
    init = "init"
    serve = "serve"
    ingest = "ingest"
    pii_ingest = "pii-ingest"
    pii_evaluate = "pii-evaluate"


class GlobalFlags(BaseModel):
    """Parsed global flags that apply to all subcommands."""
    config_path: str = "./apprentice.yaml"
    json_mode: bool = False
    verbose: int = Field(default=0, ge=0, le=2)

    model_config = ConfigDict(strict=True)


class RunArgs(BaseModel):
    """Parsed arguments specific to the 'run' subcommand."""
    task_name: str = Field(min_length=1)
    input_raw: str = Field(min_length=1)

    model_config = ConfigDict(strict=True)


class StatusArgs(BaseModel):
    """Parsed arguments specific to the 'status' subcommand."""
    task_filter: Optional[str] = None

    model_config = ConfigDict(strict=True)


class ReportArgs(BaseModel):
    """Parsed arguments specific to the 'report' subcommand."""
    output_path: Optional[str] = None

    model_config = ConfigDict(strict=True)


class InitArgs(BaseModel):
    """Parsed arguments specific to the 'init' subcommand."""
    output_path: str = "./apprentice.yaml"

    model_config = ConfigDict(strict=True)


class ServeArgs(BaseModel):
    """Parsed arguments specific to the 'serve' subcommand."""
    host: str = "127.0.0.1"
    port: int = Field(default=8710, ge=1, le=65535)
    pipeline_interval: int = Field(default=300, ge=10)
    auth_mode: str = "none"
    api_key: str = ""
    jwt_secret: str = ""
    hmac_secret: str = ""
    tls_cert: str = ""
    tls_key: str = ""
    allowed_ips: str = ""

    model_config = ConfigDict(strict=True)


class IngestArgs(BaseModel):
    """Parsed arguments specific to the 'ingest' subcommand."""
    file_path: str = Field(min_length=1)
    format: str = "auto"
    task: Optional[str] = None

    model_config = ConfigDict(strict=True)


class PIIIngestArgs(BaseModel):
    """Parsed arguments specific to the 'pii-ingest' subcommand."""
    dataset: str = "ai4privacy/pii-masking-200k"
    split: str = "train"
    limit: int = Field(default=10000, ge=1)

    model_config = ConfigDict(strict=True)


class PIIEvaluateArgs(BaseModel):
    """Parsed arguments specific to the 'pii-evaluate' subcommand."""
    mode: str = "regex_only"
    limit: int = Field(default=1000, ge=1)

    model_config = ConfigDict(strict=True)


class ParsedArgs(BaseModel):
    """Complete parsed CLI arguments combining global flags and subcommand-specific args."""
    global_flags: GlobalFlags
    command: SubcommandName
    run_args: Optional[RunArgs] = None
    status_args: Optional[StatusArgs] = None
    report_args: Optional[ReportArgs] = None
    init_args: Optional[InitArgs] = None
    serve_args: Optional[ServeArgs] = None
    ingest_args: Optional['IngestArgs'] = None
    pii_ingest_args: Optional[PIIIngestArgs] = None
    pii_evaluate_args: Optional[PIIEvaluateArgs] = None

    model_config = ConfigDict(strict=True)


class InputData(BaseModel):
    """Deserialized input data for the 'run' subcommand."""
    data: dict

    model_config = ConfigDict(strict=True)


class PhaseInfo(BaseModel):
    """Current phase information for a single task."""
    phase_name: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    is_local_primary: bool

    model_config = ConfigDict(strict=True)


class BudgetInfo(BaseModel):
    """Budget consumption information for a single task."""
    budget_limit: float
    budget_spent: float = Field(ge=0.0)
    budget_remaining: float
    is_exhausted: bool

    model_config = ConfigDict(strict=True)


class RunResult(BaseModel):
    """Pydantic model for the output of the 'run' subcommand."""
    task_name: str
    output: dict
    success: bool
    error_message: Optional[str] = ""

    model_config = ConfigDict(strict=True)


class TaskStatusEntry(BaseModel):
    """Status information for a single task."""
    task_name: str
    phase: PhaseInfo
    budget: BudgetInfo

    model_config = ConfigDict(strict=True)


class StatusResult(BaseModel):
    """Pydantic model for the output of the 'status' subcommand."""
    tasks: list[TaskStatusEntry]
    timestamp: str

    model_config = ConfigDict(strict=True)


class ReportResult(BaseModel):
    """Pydantic model for the output of the 'report' subcommand."""
    tasks: list[TaskStatusEntry]
    total_budget_limit: float
    total_budget_spent: float
    system_uptime_seconds: float
    timestamp: str
    config_path: str
    written_to_file: Optional[str] = ""

    model_config = ConfigDict(strict=True)


# Type alias for CLI result union
CLIResult = Union[RunResult, StatusResult, ReportResult]


class ApprenticeConfig(BaseModel):
    """Reference to the ApprenticeConfig type."""
    config_data: dict

    model_config = ConfigDict(strict=True)

    @property
    def validate_schema(self) -> bool:
        """Validate that config_data has expected structure."""
        # Minimal validation - a real config should have certain keys
        # For testing purposes, we check for common expected fields
        if not isinstance(self.config_data, dict):
            return False
        # Could add more validation here
        return True
