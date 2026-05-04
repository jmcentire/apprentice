# Apprentice Story Learning Enhancement

## Overview
Extend the existing Apprentice system to support multi-step story learning while maintaining backward compatibility with atomic task routing.

## Current System
- Routes requests between frontier API and local model
- Collects training data as request/response pairs
- Handles atomic exchanges effectively
- Has 2628 existing tests that must continue passing

## Enhancement Goals
Add journey-level optimization capabilities including:
- Multi-turn conversation flow handling
- Per-journey-type phase transition tracking
- Goal completion detection and measurement
- Step efficiency analysis
- Backtracking pattern recognition
- Multi-step consistency scoring

## Key Requirements
- **Backward Compatibility**: All existing atomic task routing must remain unaffected
- **Opt-in Configuration**: Story learning enabled via `story_learning_enabled: true`
- **No New Dependencies**: Work within existing Python 3.12+ and Pydantic v2 constraints
- **Test Preservation**: All 2628 existing tests must pass
- **Frozen Models**: Maintain existing model constraints

## New Components to Implement
1. **Story Model**: Represents multi-step narratives with metadata
2. **StoryStep Model**: Individual steps within a story journey
3. **StoryCollector**: Aggregates and processes story data from Chronicler
4. **JourneyEvaluator**: Analyzes journey patterns and efficiency metrics
5. **Enhanced Phase Manager**: Extended for per-journey-type tracking

## Integration Points
- Chronicler will emit multi-step event narratives
- StoryCollector processes these narratives into training data
- JourneyEvaluator provides optimization insights
- Phase manager tracks journey-specific transitions

## Success Metrics
- Journey completion rates by type
- Multi-turn consistency scores
- Step efficiency measurements
- Backtracking frequency analysis
- Goal achievement tracking