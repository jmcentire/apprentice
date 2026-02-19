# WOS Integration

Apprentice exposes HTTP endpoints for integration with the Wander Operating System (WOS).

## Endpoints

| Route | Method | Purpose | Semantics |
|-------|--------|---------|-----------|
| `/v1/events` | POST | Event ingestion | Fire-and-forget, always 200 |
| `/v1/feedback` | POST | Agent feedback | Returns `{status, match_score}` |
| `/v1/recommendations` | POST | Get recommendation | Returns `RecommendResult` or 503 |
| `/v1/skills` | GET | List skills | Returns configured tasks with phase/confidence |

## Authentication

All WOS endpoints respect the server's configured auth mode:

- **none** — No auth required (use behind firewall/VPC)
- **api-key** — `Authorization: Bearer <key>` header
- **jwt** — HS256 JWT via `Authorization: Bearer <token>`
- **hmac** — HMAC-SHA256 via `X-Signature` header

The `/health` endpoint always bypasses auth for load balancer probes.

## Data Flow

```
WOS Agent Action
       |
       v
  POST /v1/events   ──> TrainingDataStore.add_example()
       |                       |
       v                       v
  {status: "accepted"}    Background pipeline
                               |
                               v
                         Fine-tuning trigger
```

```
WOS Support Agent
       |
       v
  POST /v1/recommendations ──> Apprentice.run(skill, context)
       |                              |
       v                              v
  RecommendResult              Local or Remote model
       |
       v
  POST /v1/feedback ──> ConfidenceEngine.record_comparison()
       |
       v
  {status: "recorded", match_score: 0.85}
```

## Event Types

| Event Type | Task Mapping | Source |
|------------|-------------|--------|
| `message_sent` | `guest_response` | Chat message from agent |
| `message_received` | `guest_response` | Chat message from guest |
| `conversation_status_changed` | `ticket_triage` | Status update |
| `task_created` | `ticket_triage` | New task created |
| `task_updated` | `ticket_triage` | Task modified |
| `refund_requested` | `refund_handling` | Refund initiated |
| `refund_decided` | `refund_handling` | Refund approved/rejected |
| `booking_modified` | `booking_modification` | Booking changed |
| `agent_assigned` | `escalation_detection` | Agent assigned to conversation |

## Sample Requests

### Emit Event

```bash
curl -X POST http://localhost:8710/v1/events \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "event_type": "message_sent",
    "conversation_id": "conv-abc123",
    "agent_id": "agent-1",
    "organization_id": "org-wander",
    "timestamp": "2024-01-15T10:30:00Z",
    "payload": {"message_id": "msg-1", "text": "Hello, how can I help?"}
  }'
```

### Get Recommendation

```bash
curl -X POST http://localhost:8710/v1/recommendations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "skill": "guest_response",
    "context": {
      "conversation_id": "conv-abc123",
      "recent_messages": [{"role": "guest", "text": "When is check-in?"}]
    },
    "request_id": "req-001"
  }'
```

### Submit Feedback

```bash
curl -X POST http://localhost:8710/v1/feedback \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{
    "request_id": "req-001",
    "skill": "guest_response",
    "feedback_type": "edit",
    "edited_output": {"text": "Check-in is at 4 PM. Let me know if you need early check-in!"}
  }'
```

### List Skills

```bash
curl http://localhost:8710/v1/skills \
  -H "Authorization: Bearer YOUR_KEY"
```
