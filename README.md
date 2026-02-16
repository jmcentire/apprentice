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

```yaml
# apprentice.yaml
tasks:
  - name: classify_ticket
    description: "Classify support tickets by category and priority"
    input_schema: {text: str, metadata: dict}
    output_schema: {category: str, priority: int, confidence: float}
    evaluator: structured_match
    match_fields: [category, priority]
    confidence_thresholds:
      phase1_to_phase2: 50
      phase2_to_phase3: 0.85
      coaching_trigger: 0.70
      emergency_threshold: 0.50

remote:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  api_key_env: ANTHROPIC_API_KEY

local:
  backend: ollama
  base_model: llama3.1:8b
  fine_tune_backend: unsloth
  fine_tune_batch_size: 100
  model_dir: ./models/

sampling:
  decay_function: exponential
  min_floor: 0.02
  window_size: 100

budget:
  daily: 10.00
  weekly: 50.00
  monthly: 150.00
```

## Architecture

18 independently testable components with zero cross-dependencies:

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
| `apprentice_class` | Composition root — wires everything together |
| `cli` | Command-line interface |
| `audit_log` | Structured event logging |
| `report_generator` | Reports, metrics, and observability |

## CLI

```bash
apprentice run config.yaml              # Start the system
apprentice status config.yaml           # Show current phase, confidence, budget
apprentice report config.yaml           # Generate summary report
```

## Testing

```bash
make test        # Run all 1,372 tests
make test-quick  # Stop on first failure
```

## Built With

This project was built using [Pact](https://github.com/jmcentire/pact) — a contract-first multi-agent software engineering framework. Pact decomposed the task into 18 components, generated contracts and tests for each, then implemented them using iterative Claude Code sessions that write code, run tests, and fix failures autonomously.

## License

MIT
