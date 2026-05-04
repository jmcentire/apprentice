# Apprentice

**Train cheap models on expensive ones. Automatically. With receipts.**

Apprentice is an adaptive model distillation framework. It starts by routing every request to a frontier API (Claude, GPT, etc.), collects the outputs as training data, fine-tunes a local model, then progressively shifts traffic to it — while continuously verifying quality. The goal: replace a $15/M-token API with a $0 local model that produces equivalent results for your specific tasks.

The caller sends a request and gets a response. They don't know whether it came from Claude, a LoRA on Llama, or both. That's the point.

## When to Use Apprentice

Apprentice is for **recurring, domain-specific tasks** where a general-purpose frontier model is overkill once you have enough examples. If you're making 10 API calls a day, the cost savings don't justify the infrastructure. If you're making 10,000, they do.

Use Apprentice when:
- You have **repetitive tasks** with consistent structure (classification, extraction, routing, summarization)
- You're spending **real money on API calls** that a fine-tuned 8B model could handle
- You need **quality guarantees** — not "hope the local model is good enough" but measurable correlation scores
- You want **phased rollout** — not a hard cutover, but a gradual, data-driven transition
- The task has a **clear evaluation criteria** (exact match, structured field comparison, semantic similarity)

Don't use Apprentice when:
- Every request is unique and creative (novel writing, open-ended brainstorming)
- You need the frontier model's full reasoning capability, not a specialized subset
- Your volume is low enough that API costs don't matter
- You're already happy with prompt engineering alone

## Philosophy: Models Are Commodities, Data Is the Asset

The frontier model is a teacher, not a dependency. Apprentice treats API calls as training signals — every response is simultaneously a result and a labeled example. Over time, the local model absorbs the teacher's behavior for your specific domain. The API bill goes down. The local model gets better. The quality metrics prove it.

This is the opposite of prompt engineering. Prompt engineering optimizes the question you ask a smart model. Apprentice optimizes which model you ask, based on evidence that a cheaper one gives the same answer.

The confidence engine doesn't trust the local model by default. It earns trust through a sliding window of comparison scores. Phase transitions happen mechanically: 50 examples collected → start comparing; 0.85 correlation sustained → start routing locally. If correlation drops, traffic shifts back. No manual intervention required.

## Quick Start

```bash
pip install apprentice-ai
```

```python
from apprentice import Apprentice

app = await Apprentice.create("apprentice.yaml")

# Routing is automatic — you don't choose the model
response = await app.run("classify_ticket", {
    "text": "My payment didn't go through",
    "metadata": {"source": "email"}
})

print(response.result)   # {"category": "billing", "priority": 2}
print(response.source)   # "local" or "remote" or "dual"

await app.close()
```

## How It Works

```
Request
  |
  v
Router ──────────────────────────────────────────────────────┐
  |                                                          |
  |  Phase 1: COLD START         Phase 2: REINFORCEMENT     |  Phase 3: STEADY STATE
  |  ┌─────────────────┐        ┌──────────────────────┐    |  ┌───────────────────┐
  |  │ Remote API only  │        │ Dual: local + remote │    |  │ Local model + spot │
  |  │ Collect examples │        │ Compare via evaluator│    |  │ checks via sampler │
  |  └────────┬────────┘        └──────────┬───────────┘    |  └────────┬──────────┘
  |           │                             │                |           │
  |           v                             v                |           v
  |     Training Data              Confidence Engine         |    Sampling Scheduler
  |           │                     (rolling window)         |    (adaptive frequency)
  |           v                             │                |           │
  |     Fine-Tuning                         v                |           v
  |     Orchestrator              Phase transition?  ────────┘   Correlation check
  |           │                   (0.85 correlation)               │
  |           v                                                    v
  |     Model Validator                                     Regress? → back to Phase 2
  └──────────────────────────────────────────────────────────────────┘
```

**Three phases, all data-driven:**

1. **Cold Start** — Every request goes to the remote API. Responses are stored as training examples. After enough examples accumulate (configurable threshold), fine-tuning begins.
2. **Reinforcement** — Both models process each request. The evaluator scores local vs. remote output. A rolling window tracks correlation. When sustained correlation exceeds the threshold, the system promotes to Phase 3.
3. **Steady State** — The local model handles most traffic. An adaptive sampler periodically sends requests to both models to verify quality hasn't degraded. If it has, the system automatically regresses to Phase 2.

## PII Protection

Apprentice includes a built-in PII detection and tokenization middleware that scrubs sensitive data before it reaches models, training stores, or audit logs. The system uses a hybrid multi-modal approach that combines fast regex patterns with optional NER model inference.

Factory-built Apprentice instances fail closed for frontier egress: if the PII
middleware does not certify a request, remote model calls are blocked rather than
sent raw.

### Detection Modes

| Mode | Strategies | Latency | Dependencies |
|------|-----------|---------|-------------|
| `regex_only` (default) | Regex patterns + field heuristics | ~0.1ms | None |
| `hybrid` | Regex + field heuristics + NER model | ~50ms | `pip install apprentice-ai[ml]` |
| `ner_only` | NER model only | ~50ms | `pip install apprentice-ai[ml]` |

### What It Detects

**Regex**: Emails, phone numbers, SSNs, credit cards, IP addresses, API keys, dates of birth
**Field Heuristics**: Sensitive field names (email, phone, ssn, password, etc.) + learned patterns from prior detections
**NER Model**: Person names, locations, organizations, miscellaneous entities — unstructured PII that regex can't catch

### How It Works

```
Input data (may contain PII)
  │
  ├─ RegexDetectionStrategy        [confidence=1.0]
  ├─ FieldHeuristicStrategy        [confidence=0.9]
  └─ NERDetectionStrategy          [confidence=varies]
  │
  ▼
Merge + deduplicate (union, highest confidence wins overlaps)
  │
  ▼
Replace spans with opaque tokens → model sees __PII_EMAIL_a1b2c3__
  │
  ▼
Post-process: restore tokens → original PII for end user
```

The system learns over time — fields that repeatedly contain PII get auto-flagged, and user feedback (false positives/negatives) adjusts confidence.

### Configuration

```yaml
pii:
  enabled: true
  detection_mode: hybrid       # regex_only | hybrid | ner_only
  ner_model: dslim/bert-base-NER
  ner_device: cpu              # cpu | cuda
  ner_confidence_threshold: 0.7
  sensitive_fields:
    - email
    - phone
    - ssn
    - password
```

### Evaluation

Apprentice includes a built-in evaluation harness for measuring PII detection quality against labeled datasets:

```bash
# Ingest the ai4privacy/pii-masking-200k dataset
apprentice pii-ingest --dataset ai4privacy/pii-masking-200k --limit 1000

# Evaluate regex baseline
apprentice pii-evaluate --mode regex_only

# Evaluate hybrid (regex + NER)
apprentice pii-evaluate --mode hybrid
```

## CLI

| Command | Purpose |
|---------|---------|
| `apprentice run <config>` | Start the system (interactive or as HTTP server) |
| `apprentice serve <config>` | Start HTTP server with REST API |
| `apprentice status <config>` | Show phase, confidence, budget for each task |
| `apprentice report <config>` | Generate summary report with metrics |
| `apprentice ingest <file>` | Bulk ingest training data from file |
| `apprentice pii-ingest` | Download and ingest PII evaluation dataset |
| `apprentice pii-evaluate` | Evaluate PII detection against labeled data |

### HTTP Server Endpoints

When running `apprentice serve`, the following endpoints are available:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/v1/run` | POST | Submit a task request (core routing) |
| `/v1/status` | GET | System status |
| `/v1/status/{skill}` | GET | Per-skill phase and confidence |
| `/v1/report` | GET | Generate metrics report |
| `/v1/events` | POST | Ingest external events (fire-and-forget) |
| `/v1/feedback` | POST | Submit feedback on recommendations |
| `/v1/labeled-examples` | POST | Submit externally generated draft/final pairs |
| `/v1/constraints/check` | POST | Preflight package constraints |
| `/v1/recommendations` | POST | Request a recommendation for a skill |
| `/v1/skills` | GET | List configured skills with phase info |
| `/v1/skill-package` | GET | Inspect the loaded external skill package |
| `/v1/package-registry` | GET | Compact package/runtime registry metadata |
| `/v1/tools` | GET | Runtime-resolved package tools |
| `/v1/tools/call` | POST | Invoke supported package tools |
| `/v1/evaluators/run` | POST | Run supported package evaluators |

## Configuration

See [`examples/apprentice.yaml`](examples/apprentice.yaml) for a complete example.

```yaml
tasks:
  - name: classify_ticket
    prompt_template: "Classify: {text}"
    evaluator: structured_match
    match_fields: [category, priority]
    confidence_thresholds:
      phase2: 50        # examples before Phase 2 begins
      phase3: 0.85      # sustained correlation to enter Phase 3

  - name: extract_entities
    prompt_template: "Extract entities from: {text}"
    evaluator: semantic_similarity
    confidence_thresholds:
      phase2: 100
      phase3: 0.90

remote:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  api_key: env:ANTHROPIC_API_KEY

local:
  backend: ollama
  base_model: llama3.1:8b

budget:
  monthly_limit_usd: 150.00
  alert_threshold_pct: 80

server:
  host: 0.0.0.0
  port: 8787
```

### External Skill Packages

Apprentice's repo should not contain your product-specific ontology. Keep that
in the host application's repo or deploy-time config directory, then point
Apprentice at it from the launch config:

```yaml
skill_package:
  path: ~/MyProject/apprentice.conf.d/skill-package.yaml
  overlays:
    - ~/MyProject/apprentice.conf.d/prod.yaml
  environment: prod
```

Multiple independent packages can be composed at launch:

```yaml
skill_packages:
  - path: ~/ProjectA/apprentice.skill.yaml
  - path: ~/ProjectB/apprentice.skill.yaml
    overlays:
      - ~/ProjectB/apprentice.prod.yaml
```

Run it the same way from a shell, systemd unit, or launchd plist:

```bash
apprentice --config ~/MyProject/apprentice.yaml serve --host 127.0.0.1 --port 8710
```

The package declares skills, methods of evocation, actions, outcomes,
parameters, constraints, tools, evaluator bindings, multimodal artifacts,
feedback signals, compatibility metadata, runtime/environment bindings, and
event mappings. See
[`examples/skill-package.yaml`](examples/skill-package.yaml) for the public
shape. Private packages should live outside this repository.

Package-aware clients can inspect the loaded package with `GET
/v1/skill-package`, preflight constraints with `POST /v1/constraints/check`,
pass `method` and `action` to `POST /v1/run`, and submit externally generated
draft/final training pairs with `POST /v1/labeled-examples`. Tenant-qualified
task names such as `tenant-1:classify_ticket` resolve against the package skill
`classify_ticket`, so tenant identifiers do not belong in the package file.
`GET /v1/package-registry` and `GET /v1/tools` expose a compact runtime view for
hosts that want discovery without parsing YAML directly.

Package files can be checked before deployment:

```bash
apprentice package validate ~/MyProject/apprentice.skill.yaml \
  --overlay ~/MyProject/apprentice.prod.yaml \
  --environment prod

apprentice package inspect ~/MyProject/apprentice.skill.yaml --json
apprentice package diff ~/MyProject/apprentice.v2.yaml --compare-to ~/MyProject/apprentice.v1.yaml
```

The built-in runtime resolves credentials from environment variables referenced
by `auth_ref` and redacts them in diagnostics. Direct tool execution is
fail-closed: inert `builtin` tools may run through the built-in preflight client,
but HTTP, `cli`, `python`, and `mcp` tools require an explicit host preflight
client before execution. Unsupported tool kinds remain discoverable but are not
executed by Apprentice's built-in runtime.

When a package is loaded by the factory, Apprentice persists active registry
metadata to `.apprentice/package_registry.json`, including package ID, version,
schema version, fingerprint, environment, overlays, and validation diagnostics.
`apprentice package diff` classifies package changes as breaking, warning, or
safe to support migration review before deployment.

### Multi-Task Configuration

Each task gets its own phase progression, confidence window, and evaluator. A single Apprentice instance can manage dozens of tasks simultaneously — one might be in Phase 3 (local model proven) while another is still collecting examples in Phase 1.

## Architecture

28 components organized in two layers — 21 leaf implementations with zero cross-dependencies, wired together by 7 integration compositions:

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
| `cli` | Command-line interface and HTTP server |
| `audit_log` | Structured event logging (JSONL) |
| `report_generator` | Reports, metrics, and observability |
| `pii_tokenizer` | PII detection middleware with learned patterns |
| `pii_detection` | Multi-strategy PII detection (regex, NER, heuristic) |
| `pii_evaluation` | Span-level PII detection evaluation harness |

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

## Development

```bash
git clone https://github.com/jmcentire/apprentice.git
cd apprentice
make dev          # Install with dev + lint dependencies
make test         # Run all 2,628 tests (2,486 unit/integration + 142 smoke)
make test-quick   # Stop on first failure
make lint         # Run ruff linter
make lint-fix     # Auto-fix lint issues
make clean        # Remove build artifacts
```

The `tests/smoke/` directory contains 142 pact-generated smoke tests across 44 files, covering all 51 source modules. These tests verify every module imports correctly and every public function is callable. Generated via `pact adopt --dry-run`.

Requires Python 3.12+. Core dependencies: `pydantic`, `pyyaml`, `httpx`. Optional: `pip install apprentice-ai[ml]` for NER-based PII detection (adds `transformers`, `torch`, `datasets`).

## Built With

This project was built using [Pact](https://github.com/jmcentire/pact) — a contract-first multi-agent software engineering framework. Pact decomposed the task into 25 components, generated contracts and tests for each, then implemented them using iterative Claude Code sessions that write code, run tests, and fix failures autonomously.

## Background

Apprentice is one of three systems (alongside [Pact](https://github.com/jmcentire/pact) and Emergence) built to test the ideas in [Beyond Code: Context, Constraints, and the New Craft of Software](https://www.amazon.com/dp/B0GNLTXVC7). The book covers the coordination, verification, and specification problems that motivated these designs.

## License

MIT
