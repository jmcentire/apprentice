# Apprentice — System Context

## What It Is
Adaptive model distillation. Routes between frontier API and local fine-tuned model, progressively shifting traffic as correlation proves quality.

## How It Works
Request -> Router -> [frontier | local] -> Evaluator -> Phase Manager
Phases: shadow -> canary -> primary -> autonomous

## Key Constraints
- PII tokenized before storage (C001)
- Phase transitions require statistical validation (C002)
- Budget exhaustion degrades gracefully (C003)
- Audit log is append-only (C004)
- No global state (C005)
- New tasks start in shadow phase (C006)

## Architecture
28 components (21 leaf + 7 compositions). Core: router, phase_manager, evaluators, budget_manager, pii_tokenizer, audit_log.

## Done Checklist
- [ ] PII tokenization verified before storage
- [ ] Phase transition requires correlation threshold
- [ ] Budget exhaustion falls back to local model
- [ ] Tests pass without GPU/API/network
- [ ] Audit trail is append-only and complete
