# Apprentice

Adaptive model distillation with coaching. Start with frontier API models, progressively train a local model, then withdraw the expensive dependency — while maintaining quality guarantees.

## How It Works

Apprentice manages the full lifecycle of distilling knowledge from remote frontier models (Claude, GPT, etc.) into specialized local models:

1. **Phase 1 — Cold Start**: Every request goes to the remote API. Responses are collected as training data.
2. **Phase 2 — Reinforcement**: The local model begins attempting responses alongside the remote. Outputs are compared via the confidence engine.
3. **Phase 3 — Steady State**: The local model handles most requests. Adaptive sampling periodically checks quality against the remote, adjusting frequency based on correlation.

The caller submits a request and gets a response. They don't know whether it came from a local model, a remote API, or a blend of both.

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from apprentice import Apprentice

# Initialize from config
app = await Apprentice.create("apprentice.yaml")

# Send a request — routing is automatic
response = await app.run("classify_ticket", {
    "text": "My payment didn't go through",
    "metadata": {"source": "email"}
})

print(response.result)   # {"category": "billing", "priority": 2}
print(response.source)   # "local" or "remote" or "dual"

await app.close()
```

## Configuration

See [`examples/apprentice.yaml`](examples/apprentice.yaml) for a complete example. Key sections:

```yaml
tasks:
  - name: classify_ticket
    prompt_template: "Classify: {text}"
    evaluator: structured_match
    match_fields: [category, priority]
    confidence_thresholds:
      phase2: 50        # examples before Phase 2
      phase3: 0.85      # correlation for Phase 3

remote:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  api_key: env:ANTHROPIC_API_KEY

local:
  backend: ollama
  base_model: llama3.1:8b

budget:
  monthly_limit_usd: 150.00
```

## Architecture

25 components organized in two layers — 18 leaf implementations with zero cross-dependencies, wired together by 7 integration compositions:

### Leaf Components

| Component | Purpose |
|-----------|---------|
| `config_loader` | Load and validate YAML configuration |
| `task_registry` | Manage task type definitions and schemas |
| `data_models` | Shared Pydantic models across all components |
| `remote_api_client` | Multi-provider API abstraction (Anthropic, OpenAI, etc.) |
| `local_model_server` | Local model inference (Ollama, vLLM, llama.cpp) |
| `evaluators` | Response quality scoring (exact match, semantic, structured) |
| `phase_manager` | Phase 1/2/3 lifecycle and transitions |
| `rolling_window` | Sliding window correlation tracking |
| `sampling_scheduler` | Adaptive sampling frequency control |
| `training_data_store` | Training example collection and management |
| `fine_tuning_orchestrator` | Fine-tuning pipeline (LoRA, OpenAI, HuggingFace) |
| `model_validator` | Pre-promotion model quality validation |
| `budget_manager` | Multi-window spend tracking and enforcement |
| `router` | Request routing (local, remote, dual) |
| `apprentice_class` | Core Apprentice class — run, status, report |
| `cli` | Command-line interface |
| `audit_log` | Structured event logging (JSONL) |
| `report_generator` | Reports, metrics, and observability |

### Integration Compositions

| Composition | Children | Purpose |
|-------------|----------|---------|
| `config_and_registry` | config_loader, task_registry, data_models | Configuration + type system |
| `confidence_engine` | evaluators, phase_manager, rolling_window | Quality tracking pipeline |
| `external_interfaces` | remote_api_client, local_model_server | External service adapters |
| `training_pipeline` | training_data_store, fine_tuning_orchestrator, model_validator | Training lifecycle |
| `unified_interface` | apprentice_class, cli | User-facing API + CLI |
| `reporting` | audit_log, report_generator | Observability layer |
| `root` | all 6 compositions above | Full system composition root |

## CLI

```bash
apprentice run config.yaml              # Start the system
apprentice status config.yaml           # Show current phase, confidence, budget
apprentice report config.yaml           # Generate summary report
```

## Development

```bash
make dev         # Install with dev + lint dependencies
make test        # Run all 2,064 tests
make test-quick  # Stop on first failure
make lint        # Run ruff linter
make lint-fix    # Auto-fix lint issues
make clean       # Remove build artifacts
```

## Built With

This project was built using [Pact](https://github.com/jmcentire/pact) — a contract-first multi-agent software engineering framework. Pact decomposed the task into 25 components, generated contracts and tests for each, then implemented them using iterative Claude Code sessions that write code, run tests, and fix failures autonomously.

## License

MIT
