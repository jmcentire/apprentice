# Apprentice: Story-Level Learning

## What This Is

A targeted modification to Apprentice (adaptive model distillation) to support learning from multi-step stories instead of only atomic request/response pairs.

## Current State

Apprentice routes requests between frontier API and local model. Training data is collected as TrainingExample objects (request_id, task_type, prompt, remote_response, local_response, phase, confidence). The fine-tuning orchestrator expects single (user_prompt, assistant_response) pairs. Phase transitions are per-task-type.

Evaluators score individual responses (exact_match, semantic_similarity, structured_match, llm_judge, custom). No multi-step evaluation exists.

## What Changes

1. Add Story and StoryStep models to data_models.py
2. Add StoryCollector to training_data_store.py (store/retrieve stories, convert steps to sequential training examples)
3. Add JourneyEvaluator to evaluators.py (scores: goal_completion, step_efficiency, backtracking, consistency)
4. Extend phase manager for per-journey-type phase tracking
5. Opt-in via config: story_learning_enabled: true

## Why

When Chronicler emits stories (multi-step event narratives), Apprentice can learn from sequential patterns rather than isolated exchanges. This enables journey-level optimization — the local model learns to handle multi-turn flows, and phase transitions can vary by journey type (checkout may be autonomous while support is still coaching).

## Constraints

- Backward compatible: existing atomic task routing unaffected
- Story support opt-in via config
- No new external dependencies
- All existing 2628 tests must pass
- Python 3.12+, Pydantic v2, frozen models
