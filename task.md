# Task: Apprentice — Adaptive Model Distillation with Coaching

## Problem

Running specialized AI tasks via commercial APIs (Claude, GPT, etc.) is expensive at scale and in perpetuity. But local/fine-tuned models alone lack the quality ceiling of frontier models. What's needed is a system that **starts with the best, teaches a local model, then gradually withdraws the expensive dependency** — while maintaining quality guarantees.

## What to Build

A Python framework that presents a **single unified interface** for AI task execution. Behind that interface, the system manages the full lifecycle of distilling knowledge from remote frontier models into specialized local models, with adaptive coaching when the local model's quality drifts.

The caller submits a request and gets a response. They don't know — and shouldn't need to know — whether the response came from a local model, a remote API, or a blend of both.

## Core Lifecycle

The system progresses through three phases automatically:

### Phase 1: Cold Start (All Training)
Every request goes to the remote API. Responses are collected as training data. The local model does not serve any requests. This phase builds the initial training corpus.

### Phase 2: Reinforcement (Training with Attempts)
The local model begins attempting responses. Both local and remote models receive the same requests. The local model's outputs are compared against the remote's via the confidence engine. Training continues with the remote responses as ground truth. As correlation improves, the sampling frequency of remote requests decreases.

### Phase 3: Steady State (Local-Primary with Adaptive Coaching)
The local model handles most requests. A **variable sampling schedule** periodically sends requests to both models. The sampling frequency is determined by the running correlation (or anti-correlation) between local and remote responses across averaged series of samples:

- **High correlation** (local consistently matches remote): sampling frequency decreases toward a configurable floor (e.g., 1 in 50 requests)
- **Declining correlation**: sampling frequency increases
- **Anti-correlation** (local diverging from remote): system escalates — increases coaching intensity, may trigger retraining, alerts operator

The transition between phases and the sampling frequency within Phase 3 are **continuous and adaptive**, not discrete jumps.

## Key Design Requirements

### 1. Unified Interface
A single function/class that accepts a task request and returns a result. The caller's experience is identical regardless of which model(s) produced the answer. The interface is defined by config — task types, input/output schemas, expected behavior.

### 2. Task Registry
The system is configured with a **task list** — each task type defines:
- A name and description
- Input/output schema (what the model receives and what it should produce)
- Evaluation criteria (how to score quality — exact match, semantic similarity, structured comparison, custom evaluator)
- Confidence thresholds (when to graduate from Phase 1→2→3, when to escalate coaching)

### 3. Confidence Engine
Tracks quality over time per task type:
- Computes correlation between local and remote responses using task-specific evaluators
- Maintains a **rolling window** of comparison samples
- Derives the current sampling frequency from the correlation trend
- Detects drift (gradual degradation) and shift (sudden quality change)
- Exposes current confidence scores and phase per task type

### 4. Adaptive Sampling Scheduler
Determines which requests get sent to both models vs. local only:
- In Phase 2: all requests go to both (100% sampling)
- In Phase 3: sampling rate is a function of recent correlation
- The function is configurable (linear decay, exponential, step function, etc.)
- Has a configurable minimum sampling floor (never goes to 0% — always spot-checks)
- Can force 100% sampling temporarily when confidence drops below threshold

### 5. Training Pipeline
Manages the local model lifecycle:
- Collects training examples (input + remote response pairs)
- Triggers fine-tuning when sufficient new examples accumulate (configurable batch size)
- Supports multiple fine-tuning backends (OpenAI fine-tuning API, local LoRA via Unsloth/Axolotl, Hugging Face)
- Manages model versions — new fine-tuned model replaces old after validation
- Validates new model against a held-out test set before promotion

### 6. Local Model Server
Serves the local model for inference:
- Abstraction over Ollama, vLLM, llama.cpp, or similar
- Model loading, swapping (when a new version is promoted), health checks
- Handles the case where no local model exists yet (Phase 1)

### 7. Remote API Client
Abstraction over commercial AI APIs:
- Supports multiple providers (Anthropic, OpenAI, Google, etc.)
- Handles auth, rate limiting, retries, error recovery
- Tracks per-request cost for budget enforcement

### 8. Budget Manager
Enforces spending limits:
- Per-task, daily, weekly, monthly budget caps
- Tracks actual API spend (remote) vs. estimated savings (local)
- Reports cost trajectory — shows the system is converging toward lower spend
- Can pause remote coaching (accepting quality risk) if budget exhausted

### 9. Reporting & Observability
- Current phase per task type
- Confidence scores and trends
- Sampling frequency and correlation history
- Cost breakdown (remote vs. local compute)
- Alerts on quality degradation or budget warnings
- All metrics exportable (JSON, CLI summary, optional webhook)

## What This Is NOT

- Not a general-purpose model training framework — it's an **operational system** for a specific use case: reducing API dependency for recurring tasks
- Not a chatbot or conversational AI — tasks are defined, structured operations
- Not real-time latency-critical — casual pace is fine (seconds per response is acceptable)
- Not a model marketplace — it trains ONE local model per task type

## Configuration Interface

Everything is driven by a single config file. Example:

```yaml
# apprentice.yaml
tasks:
  - name: classify_ticket
    description: "Classify support tickets by category and priority"
    input_schema: {text: str, metadata: dict}
    output_schema: {category: str, priority: int, confidence: float}
    evaluator: structured_match  # or: semantic_similarity, exact_match, custom
    match_fields: [category, priority]  # for structured_match
    confidence_thresholds:
      phase1_to_phase2: 50  # training examples before attempting local
      phase2_to_phase3: 0.85  # correlation score to graduate
      coaching_trigger: 0.70  # correlation below this increases sampling
      emergency_threshold: 0.50  # below this, revert to full remote

  - name: extract_entities
    description: "Extract named entities from property descriptions"
    # ... similar structure

remote:
  provider: anthropic
  model: claude-sonnet-4-5-20250929
  api_key_env: ANTHROPIC_API_KEY
  # Optionally multiple providers for training diversity
  additional_providers:
    - provider: openai
      model: gpt-4o
      api_key_env: OPENAI_API_KEY

local:
  backend: ollama  # or: vllm, llama_cpp
  base_model: llama3.1:8b  # free, open-weight, LoRA-friendly (alt: mistral:7b)
  fine_tune_backend: unsloth  # or: openai, huggingface, axolotl
  fine_tune_batch_size: 100  # examples before triggering training
  model_dir: ./models/

sampling:
  decay_function: exponential  # or: linear, step
  min_floor: 0.02  # never sample less than 2% of requests
  window_size: 100  # rolling window for correlation calculation
  trend_sensitivity: 0.05  # correlation change that triggers frequency adjustment

budget:
  daily: 10.00
  weekly: 50.00
  monthly: 150.00
  currency: USD
```

## Success Criteria

1. A user can define a task in config, start the system, and send requests through the unified interface
2. The system handles Phase 1→2→3 progression automatically with zero manual intervention
3. Sampling frequency adapts to measured correlation between local and remote
4. Cost decreases over time as the local model improves (observable in reports)
5. Quality remains within configured thresholds (no silent degradation)
6. Budget limits are enforced — system degrades gracefully, doesn't overspend
7. All components have clean interfaces, are independently testable, and composable

## Constraints

- Python 3.12+
- Minimal dependencies — stdlib + pydantic + httpx for API calls
- No dependency on Pact itself — this is a standalone tool
- Must be testable without real API keys or GPU (mock everything at boundaries)
- Config-driven — behavior changes come from config, not code changes
