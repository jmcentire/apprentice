"""Example WOS client showing the event -> recommend -> feedback loop."""

import asyncio
import httpx


BASE_URL = "http://localhost:8710"
API_KEY = "your-api-key-here"

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}


async def main():
    async with httpx.AsyncClient(base_url=BASE_URL, headers=HEADERS) as client:
        # 1. Emit an event (fire-and-forget)
        event_resp = await client.post("/v1/events", json={
            "event_type": "message_received",
            "conversation_id": "conv-demo-1",
            "agent_id": "agent-42",
            "organization_id": "org-wander",
            "timestamp": "2024-01-15T10:30:00Z",
            "payload": {
                "message_id": "msg-100",
                "text": "When is check-in time?",
                "role": "GUEST",
            },
        })
        print(f"Event: {event_resp.status_code} {event_resp.json()}")

        # 2. Get a recommendation
        rec_resp = await client.post("/v1/recommendations", json={
            "skill": "guest_response",
            "context": {
                "conversation_id": "conv-demo-1",
                "recent_messages": [
                    {"role": "guest", "text": "When is check-in time?"},
                ],
            },
            "request_id": "req-demo-1",
        })
        print(f"Recommendation: {rec_resp.status_code} {rec_resp.json()}")

        recommendation = rec_resp.json()

        # 3. Submit feedback on the recommendation
        feedback_resp = await client.post("/v1/feedback", json={
            "request_id": recommendation["recommendation_id"],
            "skill": "guest_response",
            "feedback_type": "accept",
        })
        print(f"Feedback: {feedback_resp.status_code} {feedback_resp.json()}")

        # 4. List skills
        skills_resp = await client.get("/v1/skills")
        print(f"Skills: {skills_resp.status_code} {skills_resp.json()}")


if __name__ == "__main__":
    asyncio.run(main())
