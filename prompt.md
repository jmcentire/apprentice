# Apprentice: Adaptive Model Distillation Framework

## System Context

Apprentice is a production-grade adaptive model distillation framework that progressively shifts AI workloads from expensive frontier APIs to cost-effective local models while maintaining quality guarantees. The system serves multiple applications (Reeve, Wander, third-party apps) and provides HTTP REST endpoints, CLI interfaces, and package-aware client APIs. Built on the Pact framework with 28 components in a two-layer architecture: 21 leaf implementations with zero cross-dependencies, wired by 7 integration compositions.

The system operates through three distinct phases: Cold Start (100% remote API), Reinforcement (dual routing with quality comparison), and Steady State (local model with spot verification). Users include developers, CLI operators, HTTP API clients, and applications requiring budget-aware AI operations with PII protection.

## Consequence Map

**Critical Failures (Service Disruption):**
- Budget limits exceeded causing complete service interruption
- Quality correlation drops triggering automatic regression from Phase 3 to Phase 2
- PII detection false negatives exposing sensitive data to frontier APIs

**High-Impact Failures (Data/Security):**
- Source-watermark risk in frontier model responses contaminating training data
- PII detection false positives blocking legitimate requests
- Autosave system unbounded growth causing storage exhaustion

**Medium-Impact Failures (Performance):**
- Kindex rust support limitations affecting knowledge graph integration
- Tool execution blocked for legitimate cli/python/mcp operations
- Package validation failures preventing skill deployment

## Failure Archaeology

**Learned from Reeve v0.5 Integration (2026-05-03):**
- Base 42.9% vs Tense 4 diagnostic revealed correlation measurement challenges
- TalentSync v2 constraints needed 5 matches/week cap to prevent resource exhaustion
- Batch processing shipped successfully but highlighted need for better phase transition logic

**Architectural Lessons:**
- Zero cross-dependencies in leaf components prevents cascading failures
- Pact framework's single asyncio event loop assumption enables safe in-process parallelism
- 2,628 total tests (including 142 pact-generated smoke tests) provide confidence but revealed gaps in PII detection coverage

## Dependency Landscape

**External Services:** Frontier APIs (Anthropic, OpenAI), Ollama, vLLM, llama.cpp, HuggingFace
**Core Infrastructure:** SQLite database, Kindex knowledge graph, WOS, Reeve v0.5
**Libraries:** transformers, torch, datasets, pydantic, pyyaml, httpx
**Internal Components:** Exemplar components, skill packages, constraint evaluation systems

The system touches budget tracking systems, PII detection pipelines, and model training workflows. Applications depend on Apprentice for cost-effective AI operations while Apprentice depends on external APIs for frontier capabilities and local infrastructure for model serving.

## Boundary Conditions

**In Scope:** Budget-aware operations, PII protection, domain-neutral skill DSL, backward compatibility for existing endpoints, three-phase progression management, multi-task instances with independent tracking.

**Out of Scope:** Novel creative tasks, open-ended brainstorming, tasks requiring full frontier model reasoning, low-volume usage under 10,000 calls, hard-coded host application concepts in generic modules.

**Constraints:** Conservative tool execution (HTTP and builtin only), private host package content stays outside repo, runtime behavior must be inspectable/auditable/testable without live credentials.

## Success Shape

A production-grade, domain-neutral system that achieves measurable cost reduction from $15/M-token to $0 with equivalent quality. Features explicit dynamic task materialization, data-driven phase transitions, and hybrid PII detection strategies. Maintains zero cross-dependencies in leaf components while providing comprehensive audit trails and runtime inspection capabilities.

## Done When

- [ ] 10 specific AI crawlers validated for robots.txt compliance
- [ ] Product/Offer/Organization/ProductGroup extraction from rendered DOM with validation
- [ ] Skill packages support execution semantics for methods/tools/constraints
- [ ] Budget-aware operations with PII protection active
- [ ] 2815 tests passed, 19 skipped in quick suite
- [ ] Package CLI validate/inspect/diff commands functional
- [ ] Base+overlay config launch integration tests pass
- [ ] Tool preflight blocked/allowed testing verified
- [ ] Evaluator confidence update verification complete
- [ ] Package registry persistence tested
- [ ] Migration classification coverage achieved
- [ ] Three-phase progression with configurable thresholds operational
- [ ] Multi-task management with independent phase tracking verified
- [ ] PII detection evaluated against labeled datasets
- [ ] HTTP server with 15+ REST endpoints deployed
- [ ] 28 components with proper separation of concerns maintained

## Trust and Authority Model

The system operates with a base trust floor of 0.10 and authority override floor of 0.40, using decay lambda 0.05 for trust degradation. Data is classified into five tiers: PII, FINANCIAL, AUTH, COMPLIANCE, and PUBLIC. PII, FINANCIAL, AUTH, and COMPLIANCE data requires human gates and extended canary soak periods (6h-72h) before deployment. Authority is distributed across components based on data ownership domains, with conflicts resolved through trust scoring and human intervention when trust falls below authority thresholds or unresolvable conflicts arise.

## Component Topology

The system comprises 28 components in two layers: 21 leaf components handling specific functions (API routing, model management, PII detection, budget tracking, skill execution) connected through 7 integration compositions that coordinate workflows. Data flows include request routing (HTTP/gRPC), model outputs (training data), quality metrics (correlation scores), and constraint evaluations. Each component maintains explicit data access permissions and constraint bindings, with the router component serving as the primary traffic orchestrator and the evaluator component managing quality assessments and phase transitions.