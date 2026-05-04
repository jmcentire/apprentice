from decimal import Decimal
from types import SimpleNamespace

import pytest

from apprentice.factory import RouterBudgetAdapter, RouterRemoteAdapter


class FakeBudgetManager:
    def __init__(self, allowed=True):
        self.allowed = allowed
        self.authorized = []
        self.recorded = []
        self.released = []

    def authorize(self, estimated_cost):
        self.authorized.append(estimated_cost)
        return SimpleNamespace(
            allowed=self.allowed,
            remaining=Decimal("10.00") if self.allowed else Decimal("0"),
        )

    def get_report(self):
        return SimpleNamespace(total_all_time_spend=Decimal("2.00"))

    def record_spend(self, actual_cost, estimated_cost, metadata):
        self.recorded.append((actual_cost, estimated_cost, metadata))

    def release_reservation(self, estimated_cost):
        self.released.append(estimated_cost)


async def test_router_budget_adapter_reserves_then_records_spend():
    manager = FakeBudgetManager()
    adapter = RouterBudgetAdapter(manager, Decimal("0.25"))

    snapshot = await adapter.check_budget()
    await adapter.record_spend(0.10)

    assert snapshot.is_exhausted is False
    assert manager.authorized == [Decimal("0.25")]
    assert manager.recorded[0][0] == Decimal("0.1")
    assert manager.recorded[0][1] == Decimal("0.25")


async def test_router_budget_adapter_releases_reservation_for_local_only():
    manager = FakeBudgetManager()
    adapter = RouterBudgetAdapter(manager, Decimal("0.25"))

    await adapter.check_budget()
    await adapter.record_spend(0.0)

    assert manager.released == [Decimal("0.25")]
    assert manager.recorded == []


async def test_router_remote_adapter_fails_closed_without_pii_guard():
    adapter = RouterRemoteAdapter(client=object(), require_pii_guard=True)
    request = SimpleNamespace(prompt="hello", metadata={}, request_id="req-1")

    with pytest.raises(RuntimeError, match="PII middleware did not certify"):
        await adapter.call(request)
