# Task: Add Story-Level Learning to Apprentice

## Context

This project was adopted from an existing codebase (51 source files, 683 functions, 39 test files). Apprentice currently operates on atomic request/response pairs. When Chronicler starts emitting stories (multi-step event narratives), Apprentice needs to learn from sequences, not just individual exchanges.

## Specific changes

### 1. Story Model

Add to `src/apprentice/data_models.py`:

```python
class StoryStep(BaseModel):
    step_id: str
    task_type: str
    input_data: dict
    output_data: dict
    model_used: str          # "local" or "remote"
    confidence_at_step: float
    metadata: dict

class Story(BaseModel):
    story_id: str
    story_type: str          # "request", "service", "journey"
    steps: list[StoryStep]
    created_at: str
    closed_at: str
    context: dict            # Shared context across steps
```

### 2. Story Collector

Add to `src/apprentice/training_data_store.py`:
- `store_story(story: Story)` — persist to `{base_dir}/stories/{story_type}.jsonl`
- `get_story_batch(story_type, split)` — retrieve stories for training
- Convert story steps to sequential training examples: step N-1's output becomes context for step N

### 3. Journey Evaluator

Add to `src/apprentice/evaluators.py`:
- Evaluates a complete story, not just one step
- Scores: goal_completion (did the journey reach terminal state?), step_efficiency (useful steps / total), backtracking (repeated/reversed steps), consistency (outputs coherent across steps)
- Returns composite score in [0.0, 1.0]
- Implements existing EvaluatorProtocol

### 4. Per-Journey Phase Tracking

Extend `src/apprentice/phase_manager.py`:
- Support per-journey-type phase tracking (checkout_flow may be autonomous while support_flow is still coaching)
- `PhaseMetrics` gets optional `story_type` field
- `compute_phase()` accepts story_type for per-type thresholds

## Constraints

- Backward compatible: existing atomic task routing is unaffected
- Story support is opt-in via config: `story_learning_enabled: true`
- No new external dependencies
- All existing tests must pass
- Python 3.12+, Pydantic v2
