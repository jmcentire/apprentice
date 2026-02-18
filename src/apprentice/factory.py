"""Component Factory for Apprentice.

Single async function `build_from_config` that constructs all real components
in dependency order, creates adapter classes for the Router's Protocol-based DI,
and returns a fully wired Apprentice instance.
"""

import os
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


# ============================================================================
# Router Adapter Classes
# ============================================================================


class RouterBudgetAdapter:
    """Adapts BudgetManager to the Router's BudgetManager protocol."""

    def __init__(self, budget_manager: Any, cost_per_request: Decimal):
        self._bm = budget_manager
        self._cost_per_request = cost_per_request

    async def check_budget(self) -> Any:
        from apprentice.router import BudgetSnapshot
        from apprentice.budget_manager import SpendMetadata

        remaining = self._bm.remaining_budget()
        is_exhausted = remaining <= Decimal("0")
        report = self._bm.get_report()
        total = float(remaining + report.total_all_time_spend)
        return BudgetSnapshot(
            total_budget_usd=total,
            spent_usd=float(report.total_all_time_spend),
            remaining_usd=float(remaining),
            is_exhausted=is_exhausted,
        )

    async def record_spend(self, cost: float) -> None:
        from apprentice.budget_manager import SpendMetadata

        if cost <= 0:
            return
        metadata = SpendMetadata(
            request_id=str(uuid.uuid4()),
            model_name="remote",
            input_tokens=0,
            output_tokens=0,
            task_type="inference",
            timestamp=datetime.now(timezone.utc),
        )
        self._bm.record_spend(
            actual_cost=Decimal(str(cost)),
            estimated_cost=self._cost_per_request,
            metadata=metadata,
        )


class RouterConfidenceAdapter:
    """Adapts ConfidenceEngine to the Router's ConfidenceEngine protocol."""

    def __init__(self, engine: Any, task_names: list[str], training_data_store: Any, ollama_client: Any):
        self._engine = engine
        self._task_names = task_names
        self._tds = training_data_store
        self._ollama = ollama_client
        # Shared state: last snapshot for SamplingAdapter to read
        self._last_snapshot = None

    async def get_phase_context(self, task_type: str) -> Any:
        from apprentice.router import PhaseContext, Phase

        # Get real example count from training data store
        example_count = 0
        try:
            counts = self._tds.get_example_count(task_type)
            example_count = counts.total
        except Exception:
            pass

        # Check real local model availability
        local_available = True
        try:
            health = await self._ollama.health()
            local_available = health.is_ready
        except Exception:
            local_available = False

        snapshot = self._engine.get_snapshot(
            task_id=task_type,
            example_count=example_count,
            local_model_available=local_available,
        )
        self._last_snapshot = snapshot

        phase_map = {
            "REMOTE_ONLY": Phase.PHASE_1,
            "COACHING": Phase.PHASE_2,
            "COACHING_FULL": Phase.PHASE_2,
            "LOCAL_PRIMARY": Phase.PHASE_3,
            "LOCAL_ONLY": Phase.PHASE_3,
        }
        phase = phase_map.get(snapshot.phase, Phase.PHASE_1)
        confidence = snapshot.correlation_score if snapshot.correlation_score is not None else 0.0
        return PhaseContext(phase=phase, confidence=confidence)


class RouterSamplingAdapter:
    """Adapts AdaptiveSamplingScheduler to the Router's SamplingScheduler protocol."""

    def __init__(self, scheduler: Any, confidence_adapter: "RouterConfidenceAdapter", budget_adapter: "RouterBudgetAdapter"):
        self._scheduler = scheduler
        self._confidence_adapter = confidence_adapter
        self._budget_adapter = budget_adapter

    async def should_coach(self) -> bool:
        from apprentice.sampling_scheduler import ConfidenceSnapshot as SSSnapshot

        # Use real confidence data from the last get_phase_context call
        last = self._confidence_adapter._last_snapshot
        correlation = 0.0
        is_stale = False
        sample_count = 0
        if last is not None:
            correlation = last.correlation_score if last.correlation_score is not None else 0.0
            is_stale = last.is_stale
            sample_count = last.sample_count

        # Check real budget status
        budget_exhausted = False
        try:
            remaining = self._budget_adapter._bm.remaining_budget()
            budget_exhausted = remaining <= Decimal("0")
        except Exception:
            pass

        snapshot = SSSnapshot(
            correlation_score=correlation,
            is_stale=is_stale,
            sample_count=sample_count,
            budget_exhausted=budget_exhausted,
        )
        decision = self._scheduler.should_sample(snapshot)
        return decision.is_coaching_sample


class RouterLocalAdapter:
    """Adapts OllamaClient to the Router's LocalModelBackend protocol."""

    def __init__(self, ollama: Any, model_name: str):
        self._ollama = ollama
        self._model_name = model_name

    async def call(self, request: Any) -> Any:
        from apprentice.local_model_server import InferenceRequest, InferenceSuccess
        from apprentice.router import ModelResponse

        infer_req = InferenceRequest(
            prompt=request.prompt,
            model_id=self._model_name,
            request_id=request.request_id,
        )
        result = await self._ollama.infer(infer_req)

        if isinstance(result, InferenceSuccess):
            return ModelResponse(
                content=result.text,
                model_id=result.model_id,
                latency_ms=result.latency_ms,
                cost=0.0,
                token_count=result.tokens_generated,
            )
        else:
            raise RuntimeError(f"Local model unavailable: {result.reason.value}")


class RouterRemoteAdapter:
    """Adapts BaseRemoteClient to the Router's RemoteModelBackend protocol."""

    def __init__(self, client: Any):
        self._client = client

    async def call(self, request: Any) -> Any:
        from apprentice.remote_api_client import PromptMessage, MessageRole, GenerationParams
        from apprentice.router import ModelResponse

        messages = [PromptMessage(role=MessageRole.user, content=request.prompt)]
        response = await self._client.send_request(messages, GenerationParams())
        return ModelResponse(
            content=response.content,
            model_id=response.model,
            latency_ms=response.latency_ms,
            cost=response.cost_usd,
            token_count=response.usage.input_tokens + response.usage.output_tokens,
        )


class RouterTrainingAdapter:
    """Adapts TrainingDataStore to the Router's TrainingDataCollector protocol."""

    def __init__(self, store: Any):
        self._store = store

    async def collect(self, example: Any) -> None:
        from apprentice.training_data_store import TrainingExample as TDSExample

        content = ""
        if example.remote_response:
            content = example.remote_response.content
        elif example.local_response:
            content = example.local_response.content

        if not content:
            return

        tds_example = TDSExample(
            input=example.prompt,
            output=content,
            task_type=example.task_type,
            timestamp=example.collected_at_utc,
            metadata={
                "request_id": example.request_id,
                "example_type": example.example_type.value,
                "phase": example.phase.value,
            },
        )
        self._store.add_example(tds_example)


class RouterAuditAdapter:
    """Adapts JsonLinesAuditLogger to the Router's AuditLogger protocol."""

    def __init__(self, audit_logger: Any):
        self._logger = audit_logger

    async def log(self, entry: Any) -> None:
        from apprentice.audit_log import AuditEntry, EventType

        audit_entry = AuditEntry(
            task_name=entry.task_type,
            event_type=EventType.request_routed,
            details={
                "request_id": entry.request_id,
                "phase": entry.phase.value,
                "confidence": entry.confidence,
                "intended_action": entry.intended_action.value,
                "actual_action": entry.actual_action.value,
                "total_cost_usd": entry.total_cost_usd,
                "total_latency_ms": entry.total_latency_ms,
                "is_degraded": entry.is_degraded,
                "is_fallback": entry.is_fallback,
            },
        )
        self._logger.log(audit_entry)


# ============================================================================
# Factory Function
# ============================================================================


async def build_from_config(config_path: str) -> Any:
    """Build a fully wired Apprentice instance from a config file.

    Constructs each real component in DAG order, creates adapter classes
    for the Router's Protocol-based DI, and returns a ready-to-use Apprentice.
    """
    from apprentice import config_loader
    from apprentice.apprentice_class import (
        Apprentice,
        ApprenticeConfig as ACConfig,
        BudgetEnforcementConfig,
        TaskConfig as ACTaskConfig,
        ConfidenceThresholds as ACThresholds,
    )
    from apprentice.plugin_registry import PluginRegistrySet
    from apprentice.audit_log import AuditConfig, JsonLinesAuditLogger
    from apprentice.budget_manager import BudgetConfig, BudgetManager, PeriodLimit, PeriodType
    from apprentice.confidence_engine import ConfidenceEngine, ConfidenceEngineConfig
    from apprentice.evaluators import ExactMatchEvaluator, StructuredMatchEvaluator
    from apprentice.config_loader import FinetuningBackendName
    from apprentice.fine_tuning_orchestrator import (
        FineTuningOrchestrator,
        KubernetesLoRAOrchestratorConfig,
        LocalNoOpBackend,
        LocalNoOpOrchestratorConfig,
        ModelVersionStore,
        OrchestratorConfig,
    )
    from apprentice.local_model_server import OllamaClient, OllamaClientConfig
    from apprentice.remote_api_client import RemoteClientConfig, ProviderName, create_remote_client
    from apprentice.router import RequestRouter, RouterConfig
    from apprentice.sampling_scheduler import AdaptiveSamplingScheduler, SamplingSchedulerConfig
    from apprentice.task_registry import (
        TaskConfig as TRTaskConfig,
        ConfidenceThresholds as TRThresholds,
        TaskRegistry,
        EvaluatorType as TREvaluatorType,
    )
    from apprentice.training_data_store import TrainingDataStore, TrainingDataStoreConfig

    # 1. Load validated config
    cfg = config_loader.load_config(Path(config_path))

    # 1.5. Construct Plugin Registry
    plugin_registry_set = PluginRegistrySet.with_defaults()
    if cfg.plugins:
        plugin_registry_set.register_from_config(
            {domain: {name: {"factory": entry.factory} for name, entry in plugins.items()}
             for domain, plugins in cfg.plugins.items()}
        )

    # 2. Create directories
    base_dir = Path(".apprentice")
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "models").mkdir(exist_ok=True)
    (base_dir / "training_data").mkdir(exist_ok=True)
    (base_dir / "confidence").mkdir(exist_ok=True)

    audit_log_path = Path(cfg.audit.log_path)
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    budget_state_path = Path(cfg.budget.budget_state_path)
    budget_state_path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Construct components in DAG order

    # (1) Audit Logger
    audit_logger = JsonLinesAuditLogger(AuditConfig(log_path=str(audit_log_path)))

    # (2) Task Registry
    eval_type_map = {
        "exact_match": TREvaluatorType.exact_match,
        "semantic_similarity": TREvaluatorType.semantic_similarity,
        "structured_match": TREvaluatorType.structured_match,
    }
    task_configs = []
    for task in cfg.tasks:
        match_fields = []
        for ev in task.evaluators:
            for mf in ev.match_fields:
                if mf.name not in match_fields:
                    match_fields.append(mf.name)

        input_props = {f.name: {"type": f.type} for f in task.input_schema}
        output_props = {f.name: {"type": f.type} for f in task.output_schema}

        tr_task = TRTaskConfig(
            name=task.task_name,
            description=f"Task: {task.task_name}",
            input_schema={"type": "object", "properties": input_props},
            output_schema={"type": "object", "properties": output_props},
            system_prompt="You are a helpful assistant.",
            prompt_template=task.prompt_template,
            evaluator_type=eval_type_map.get(task.evaluators[0].type.value, TREvaluatorType.structured_match),
            match_fields=match_fields,
            confidence_thresholds=TRThresholds(
                phase1_to_phase2_example_count=task.min_training_examples,
                phase2_to_phase3_correlation=task.thresholds.local_only,
                coaching_trigger=task.thresholds.local_ready,
                emergency_threshold=task.thresholds.degraded_threshold,
            ),
        )
        task_configs.append(tr_task)

    task_registry = TaskRegistry(task_list=task_configs)

    # (3) Budget Manager
    period_limits = [
        PeriodLimit(period_type=PeriodType.daily, limit_amount=Decimal(str(cfg.budget.max_daily_cost_usd))),
        PeriodLimit(period_type=PeriodType.monthly, limit_amount=Decimal(str(cfg.budget.max_monthly_cost_usd))),
    ]
    budget_manager = BudgetManager(BudgetConfig(
        period_limits=period_limits,
        timezone="UTC",
        estimated_cost_per_input_token=Decimal(str(cfg.budget.cost_per_input_token)),
        estimated_cost_per_output_token=Decimal(str(cfg.budget.cost_per_output_token)),
        state_file_path=budget_state_path,
    ))

    # (4) Training Data Store
    tds = TrainingDataStore()
    tds.initialize(TrainingDataStoreConfig(
        base_dir=cfg.training_data.storage_dir,
        holdout_percentage=0.10,
        fine_tune_batch_size=cfg.finetuning.batch_size,
    ))

    # (5) Evaluators registry
    evaluator_registry = {}
    for task in cfg.tasks:
        for ev_cfg in task.evaluators:
            ev_type = ev_cfg.type.value
            if ev_type not in evaluator_registry:
                if ev_type == "exact_match":
                    evaluator_registry[ev_type] = ExactMatchEvaluator()
                else:
                    evaluator_registry[ev_type] = StructuredMatchEvaluator()

    # (6) Remote Client
    api_key_env_var = "ANTHROPIC_API_KEY"
    if api_key_env_var not in os.environ:
        try:
            os.environ[api_key_env_var] = cfg.provider.api_key.get_secret_value()
        except Exception:
            pass

    remote_client = create_remote_client(RemoteClientConfig(
        provider=ProviderName.anthropic,
        model=cfg.provider.model,
        api_key_env_var=api_key_env_var,
    ))

    # (7) Ollama Client
    ollama_client = OllamaClient(OllamaClientConfig(base_url=cfg.local_model.endpoint))

    # (8) Confidence Engine
    task_eval_configs = {}
    for task in cfg.tasks:
        ev_type = task.evaluators[0].type.value
        mfields = [mf.name for ev in task.evaluators for mf in ev.match_fields]
        task_eval_configs[task.task_name] = {
            "evaluator_type": ev_type,
            "params": {"match_fields": mfields},
        }

    confidence_engine = ConfidenceEngine(
        ConfidenceEngineConfig(
            task_evaluator_configs=task_eval_configs,
            phase_thresholds={
                "phase1_to_phase2_example_count": cfg.tasks[0].min_training_examples,
                "phase2_to_phase3_correlation": cfg.tasks[0].thresholds.local_only,
                "coaching_trigger": cfg.tasks[0].thresholds.local_ready,
                "emergency_threshold": cfg.tasks[0].thresholds.degraded_threshold,
                "sustained_windows": 3,
                "cooldown_windows": 0,
            },
            persistence_dir=str(base_dir / "confidence"),
        ),
        evaluator_registry,
    )

    # (9) Sampling Scheduler
    sampling_scheduler = AdaptiveSamplingScheduler(SamplingSchedulerConfig())

    # (10) Fine-Tuning Backend + Orchestrator + Version Store
    ft_version_store = ModelVersionStore(store_path=str(base_dir / "model_versions.json"))

    if cfg.finetuning.backend == FinetuningBackendName.kubernetes_lora:
        from apprentice.kubernetes_lora_backend import (
            KubernetesLoRABackend,
            KubernetesLoRABackendConfig,
        )
        ft_backend = KubernetesLoRABackend(KubernetesLoRABackendConfig(
            gcs_bucket=cfg.finetuning.gcs_bucket,
            training_image=cfg.finetuning.training_image,
            namespace=cfg.finetuning.k8s_namespace or "default",
            gpu_type=cfg.finetuning.gpu_type or "nvidia-tesla-t4",
            service_account_name=cfg.finetuning.service_account or "",
            base_model=cfg.finetuning.model_base,
            ollama_base_url=cfg.local_model.endpoint,
            local_gguf_dir=cfg.finetuning.output_dir,
        ))
        orch_backend_cfg = KubernetesLoRAOrchestratorConfig()
    else:
        ft_backend = LocalNoOpBackend(output_dir=cfg.finetuning.output_dir)
        orch_backend_cfg = LocalNoOpOrchestratorConfig()

    ft_orchestrator = FineTuningOrchestrator(
        backend=ft_backend,
        model_version_store=ft_version_store,
        config=OrchestratorConfig(
            backend=orch_backend_cfg,
            min_training_examples=cfg.finetuning.batch_size,
            model_version_store_path=str(base_dir / "model_versions.json"),
        ),
    )

    # (12) Build Router with adapters
    cost_per_request = (
        Decimal(str(cfg.budget.cost_per_input_token)) * 1000
        + Decimal(str(cfg.budget.cost_per_output_token)) * 500
    )
    budget_adapter = RouterBudgetAdapter(budget_manager, cost_per_request)
    confidence_adapter = RouterConfidenceAdapter(
        confidence_engine, [t.task_name for t in cfg.tasks], tds, ollama_client,
    )
    sampling_adapter = RouterSamplingAdapter(sampling_scheduler, confidence_adapter, budget_adapter)
    router = RequestRouter(
        config=RouterConfig(),
        budget_manager=budget_adapter,
        confidence_engine=confidence_adapter,
        sampling_scheduler=sampling_adapter,
        local_model_backend=RouterLocalAdapter(ollama_client, cfg.local_model.model_name),
        remote_model_backend=RouterRemoteAdapter(remote_client),
        training_data_collector=RouterTrainingAdapter(tds),
        audit_logger=RouterAuditAdapter(audit_logger),
    )

    # (13) Build Apprentice
    ac_tasks = []
    for task in cfg.tasks:
        ac_tasks.append(ACTaskConfig(
            task_name=task.task_name,
            remote_model_id=cfg.provider.model,
            local_model_id=cfg.local_model.model_name,
            budget_limit_usd=cfg.budget.max_monthly_cost_usd,
            budget_period_days=30,
            confidence_thresholds=ACThresholds(
                bootstrapping_upper=0.2,
                active_learning_upper=0.5,
                supervised_upper=0.8,
                autonomous_lower=0.9,
            ),
        ))

    ac_config = ACConfig(
        tasks=ac_tasks,
        remote_api_base_url=cfg.provider.api_base_url,
        local_model_endpoint=cfg.local_model.endpoint,
        budget_enforcement=BudgetEnforcementConfig(
            hard_limit=True, warning_threshold_pct=80.0, log_every_spend=True,
        ),
        audit_log_path=cfg.audit.log_path,
        retry_max_attempts=cfg.provider.max_retries,
        retry_backoff_base_seconds=cfg.provider.retry_base_delay_seconds,
    )

    apprentice = Apprentice(
        config=ac_config,
        config_loader=cfg,
        task_registry=task_registry,
        remote_client=remote_client,
        local_client=ollama_client,
        confidence_engine=confidence_engine,
        sampling_scheduler=sampling_scheduler,
        budget_manager=budget_manager,
        training_data_store=tds,
        router=router,
        audit_log=audit_logger,
    )

    # (14) Model Validator
    from apprentice.model_validator import ModelValidator, ValidationConfig
    model_validator = ModelValidator(
        config=ValidationConfig(),
        evaluators=list(evaluator_registry.values()),
        training_data_store=tds,
        model_server=ollama_client,
        audit_logger=audit_logger,
    )

    # Attach extra components for serve daemon access
    apprentice._ft_orchestrator = ft_orchestrator
    apprentice._ft_version_store = ft_version_store
    apprentice._model_validator = model_validator
    apprentice._real_config = cfg
    apprentice._plugin_registry_set = plugin_registry_set

    # Construct middleware pipeline if configured
    if cfg.middleware:
        from apprentice.middleware import MiddlewarePipeline
        middleware_registry = plugin_registry_set.get_registry("middleware")
        apprentice._middleware_pipeline = MiddlewarePipeline.from_config(
            cfg.middleware, middleware_registry,
        )
    else:
        apprentice._middleware_pipeline = None

    # Construct feedback collector if configured
    if cfg.feedback and cfg.feedback.enabled:
        from apprentice.feedback_collector import FeedbackCollector
        apprentice._feedback_collector = FeedbackCollector(
            storage_dir=cfg.feedback.storage_dir,
            enabled=True,
        )
    else:
        apprentice._feedback_collector = None

    # Construct observer if configured
    if cfg.observer and cfg.observer.enabled:
        from apprentice.observer import Observer, ObserverConfig as ObsCfg
        apprentice._observer = Observer(ObsCfg(
            enabled=True,
            context_window_size=cfg.observer.context_window_size,
            shadow_recommendation_rate=cfg.observer.shadow_recommendation_rate,
            min_context_before_recommending=cfg.observer.min_context_before_recommending,
        ))
    else:
        apprentice._observer = None

    # Store mode
    apprentice._mode = cfg.mode

    return apprentice
