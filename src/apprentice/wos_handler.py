"""WOS Integration Handler.

Bridge between HTTP endpoints and Apprentice internals.
Maps WOS events, feedback, and recommendations to Apprentice's core APIs.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .wos_models import (
    EVENT_TYPE_TO_TASK,
    FEEDBACK_SCORES,
    EventResponse,
    FeedbackRecord,
    FeedbackResponse,
    RecommendRequest,
    RecommendResult,
    SkillInfo,
    WOSEvent,
)

logger = logging.getLogger(__name__)


async def handle_event(apprentice: Any, event: WOSEvent) -> EventResponse:
    """Ingest a WOS event as training signal.

    Maps event types to task_ids and stores as training data.
    Fire-and-forget: always returns 200 even if handler errors internally.
    """
    try:
        task_id = EVENT_TYPE_TO_TASK.get(event.event_type.value, "ticket_triage")

        # Log to audit log if available
        if hasattr(apprentice, '_audit_log') and hasattr(apprentice._audit_log, 'log'):
            try:
                await apprentice._audit_log.log({
                    "type": "wos_event",
                    "event_type": event.event_type.value,
                    "conversation_id": event.conversation_id,
                    "task_id": task_id,
                    "timestamp": event.timestamp,
                })
            except Exception:
                pass

        # Store as training data if store is available
        tds = getattr(apprentice, '_training_data_store', None)
        if tds is not None and hasattr(tds, 'add_example'):
            try:
                from .training_data_store import TrainingExample
                example = TrainingExample(
                    input=f"{event.event_type.value}: {event.conversation_id}",
                    output=str(event.payload),
                    task_type=task_id,
                    timestamp=event.timestamp or datetime.now(timezone.utc).isoformat(),
                    metadata={
                        "source": "wos_event",
                        "agent_id": event.agent_id,
                        "organization_id": event.organization_id,
                        "conversation_id": event.conversation_id,
                    },
                )
                tds.add_example(example)
            except Exception as e:
                logger.warning(f"Failed to store WOS event as training data: {e}")

        logger.info(f"WOS event ingested: {event.event_type.value} -> {task_id}")

    except Exception as e:
        # Fire-and-forget: log but don't propagate
        logger.warning(f"WOS event handler error (non-fatal): {e}")

    return EventResponse(status="accepted")


async def handle_feedback(apprentice: Any, feedback: FeedbackRecord) -> FeedbackResponse:
    """Record agent feedback on a recommendation.

    Converts feedback_type to a numeric score and feeds into the confidence engine.
    """
    score = FEEDBACK_SCORES.get(feedback.feedback_type.value, 0.5)
    task_id = feedback.skill

    # Record via confidence engine if available
    ce = getattr(apprentice, '_confidence_engine', None)
    if ce is not None and hasattr(ce, 'record_comparison'):
        try:
            ce.record_comparison(
                task_id=task_id,
                input_data=f"feedback:{feedback.request_id}",
                local_output=str(feedback.edited_output or {}),
                remote_output=f"score:{score}",
            )
        except Exception as e:
            logger.warning(f"Failed to record feedback comparison: {e}")

    logger.info(f"feedback recorded: {feedback.skill} = {feedback.feedback_type.value} ({score})")

    return FeedbackResponse(status="recorded", match_score=score)


async def handle_recommend(apprentice: Any, request: RecommendRequest) -> RecommendResult:
    """Delegate a recommendation request to Apprentice.run().

    Maps the skill name to a task_name and calls the core run() method.
    """
    task_name = request.skill
    request_id = request.request_id or str(uuid.uuid4())

    response = await apprentice.run(task_name, request.context)

    confidence = 0.0
    if hasattr(apprentice, '_confidence_engine'):
        ce = apprentice._confidence_engine
        if hasattr(ce, 'get_snapshot'):
            try:
                snap = ce.get_snapshot(
                    task_id=task_name,
                    example_count=0,
                    local_model_available=True,
                )
                confidence = snap.correlation_score or 0.0
            except Exception:
                pass

    return RecommendResult(
        skill_name=task_name,
        recommendation_id=request_id,
        confidence=confidence,
        output=response.output if hasattr(response, 'output') else {},
    )


async def list_skills(apprentice: Any) -> list[SkillInfo]:
    """List all configured skills with their current phase and confidence."""
    skills: list[SkillInfo] = []

    registry = getattr(apprentice, '_task_registry', None)
    if registry is None:
        return skills

    task_names: list[str] = []
    if hasattr(registry, 'task_names'):
        task_names = sorted(registry.task_names)
    elif hasattr(registry, 'list_tasks'):
        task_names = sorted(registry.list_tasks())

    for name in task_names:
        phase = "bootstrapping"
        confidence: Optional[float] = None

        ce = getattr(apprentice, '_confidence_engine', None)
        if ce is not None and hasattr(ce, 'get_snapshot'):
            try:
                snap = ce.get_snapshot(
                    task_id=name,
                    example_count=0,
                    local_model_available=True,
                )
                phase = snap.phase
                confidence = snap.correlation_score
            except Exception:
                pass

        skills.append(SkillInfo(name=name, phase=phase, confidence=confidence))

    return skills
