# Apprentice

Adaptive model distillation. Routes requests between frontier API and local fine-tuned model, progressively shifting traffic as correlation proves quality. Budget-aware with PII protection.

## Quick Reference

```bash
apprentice init <config.yaml>         # initialize from config
apprentice route <task> <input>       # route a single request
apprentice train <task>               # trigger fine-tuning for task
apprentice evaluate <task>            # evaluate local model quality
apprentice report [--format json]     # phase/cost/quality report
apprentice serve [--port 8080]        # HTTP API server
python3 -m pytest tests/ -v          # run tests (2,628)
make test                             # same via Makefile
make lint                             # ruff linting
```

## Architecture

### Traffic Routing
```
Request -> Router -> [frontier API | local model] -> Response
                        |
                   Evaluator (correlation check)
                        |
                   Phase Manager (shadow -> canary -> primary -> autonomous)
```

### Phase Progression
| Phase | Behavior | Transition Trigger |
|-------|----------|-------------------|
| shadow | 100% frontier, local runs in background | Sufficient training data |
| canary | Small % to local, compare quality | Correlation above threshold |
| primary | Majority to local, frontier as fallback | Sustained high correlation |
| autonomous | 100% local | Manual or budget-triggered |

### Component Architecture (28 components)
21 leaf implementations with zero cross-dependencies + 7 integration compositions.

**Build order:**
1. config_and_registry (config_loader + task_registry + data_models)
2. confidence_engine (evaluators + phase_manager + rolling_window)
3. external_interfaces (remote_api_client + local_model_server)
4. training_pipeline (training_data_store + fine_tuning_orchestrator + model_validator)
5. unified_interface (apprentice_class + cli)
6. reporting (audit_log + report_generator)
7. root (all compositions)

## Structure

```
src/apprentice/
  config_loader.py           # YAML config parsing
  task_registry.py           # Task definitions and routing rules
  data_models.py             # Pydantic models
  remote_api_client.py       # Frontier API client (httpx)
  local_model_server.py      # Local model interface
  evaluators.py              # Quality evaluation (correlation, metrics)
  phase_manager.py           # Phase state machine
  rolling_window.py          # Sliding window statistics
  sampling_scheduler.py      # Training data sampling strategy
  training_data_store.py     # PII-scrubbed training data persistence
  fine_tuning_orchestrator.py # Fine-tuning pipeline coordination
  model_validator.py         # Model quality validation
  budget_manager.py          # API spend tracking and caps
  router.py                  # Request routing logic
  apprentice_class.py        # Main orchestrator
  cli.py                     # Click CLI
  audit_log.py               # Append-only JSONL audit (PACT:04bd77:audit_log)
  report_generator.py        # Phase/cost/quality reports
  event_handler.py           # Event dispatch (file, stdout, socket, HTTP)
  observer.py                # Action observation (rolling context window)
  pii_tokenizer.py           # PII tokenization
  pii_detection.py           # PII pattern detection (PACT:pii_detection)
  pii_evaluation.py          # PII detection quality scoring
  wos_handler.py             # WOS integration handler
  wos_models.py              # WOS data models
  kubernetes_lora_backend.py # GKE LoRA fine-tuning backend
```

## Audit Events

9 structured event types (JSONL, UTC timestamps, frozen Pydantic):
request_routed, training_example_stored, fine_tune_started, fine_tune_completed, model_validated, model_promoted, phase_transition, budget_warning, confidence_alert

## Conventions

- Python 3.12+, Pydantic v2, Click, httpx, hatchling, pytest
- No global state — explicit parameter passing
- Dependency injection for all external boundaries (API, model server, I/O)
- PII tokenization before any training data storage
- PACT keys embedded in source for traceability
- Tests: 2,628 total (81 component + 44 smoke + 142 Pact-generated), no GPU/API/network required
- CI/CD: GitHub Actions (lint + test on push, PyPI OIDC publish on tag)
- Keep files under 300 lines
- Structured JSON audit logging (append-only)

## Kindex

Apprentice captures discoveries, decisions, and distillation rationale in [Kindex](~/Code/kindex). Search before adding. Link related concepts.
