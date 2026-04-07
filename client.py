"""
vishwakarma_env/client.py

OpenEnv-compatible client for the Vishwakarma environment.
Install and use:
    from vishwakarma_env import VishwakarmaEnv, VishwakarmaAction

    async with VishwakarmaEnv(base_url="https://...hf.space") as env:
        obs = await env.reset()
        result = await env.step(VishwakarmaAction(directive="run_normal"))
"""

import asyncio
import httpx
from dataclasses import asdict
from typing import Optional, AsyncIterator

from .models import (
    VishwakarmaAction, VishwakarmaObservation,
    VishwakarmaState, MachineStatus, Alert
)


class VishwakarmaEnv:
    """
    HTTP client for the Vishwakarma Factory Environment.
    Follows the OpenEnv client pattern: async context manager,
    reset() / step() / state() methods.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    # ─────────────────────────────────────────
    # Context manager
    # ─────────────────────────────────────────

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def sync(self):
        """Return synchronous wrapper."""
        return SyncVishwakarmaEnv(self.base_url, self.timeout)

    # ─────────────────────────────────────────
    # Core API
    # ─────────────────────────────────────────

    async def reset(self) -> VishwakarmaObservation:
        resp = await self._client.post(f"{self.base_url}/reset")
        resp.raise_for_status()
        return self._parse_obs(resp.json())

    async def step(self, action: VishwakarmaAction) -> VishwakarmaObservation:
        payload = {
            "directive": action.directive,
            "reroute_from": action.reroute_from,
            "reroute_to": action.reroute_to,
            "call_maintenance": action.call_maintenance,
            "order_stock_tons": action.order_stock_tons,
            "order_stock_supplier": action.order_stock_supplier,
            "authorize_overtime_workers": action.authorize_overtime_workers,
            "call_contractors": action.call_contractors,
            "adjust_production_rate": action.adjust_production_rate,
            "accept_emergency_order": action.accept_emergency_order,
            "reasoning": action.reasoning,
        }
        resp = await self._client.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        return self._parse_obs(resp.json())

    async def state(self) -> dict:
        resp = await self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    # ─────────────────────────────────────────
    # Parser
    # ─────────────────────────────────────────

    def _parse_obs(self, data: dict) -> VishwakarmaObservation:
        machines = [
            MachineStatus(
                name=m["name"],
                online=m["online"],
                utilization=m["utilization"],
                breakdown_eta_mins=m.get("breakdown_eta_mins", 0)
            )
            for m in data.get("machines", [])
        ]
        alerts = [
            Alert(
                crisis_type=a["crisis_type"],
                message=a["message"],
                severity=a["severity"],
                resolution_options=a["resolution_options"]
            )
            for a in data.get("active_alerts", [])
        ]
        return VishwakarmaObservation(
            machines=machines,
            workers_present=data["workers_present"],
            workers_total=data["workers_total"],
            stock_tons=data["stock_tons"],
            stock_days_remaining=data["stock_days_remaining"],
            units_produced_today=data["units_produced_today"],
            units_target_today=data["units_target_today"],
            production_rate_per_hour=data["production_rate_per_hour"],
            cost_today_INR=data["cost_today_INR"],
            budget_today_INR=data["budget_today_INR"],
            cumulative_cost_INR=data["cumulative_cost_INR"],
            cumulative_budget_INR=data["cumulative_budget_INR"],
            active_alerts=alerts,
            crisis_type=data["crisis_type"],
            crisis_severity=data["crisis_severity"],
            reward=data["reward"],
            reward_breakdown=data["reward_breakdown"],
            done=data["done"],
            step=data["step"],
            shift=data["shift"],
            day=data["day"],
            feedback=data["feedback"]
        )


# ─────────────────────────────────────────────
# Synchronous wrapper
# ─────────────────────────────────────────────

class SyncVishwakarmaEnv:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self._async_env = VishwakarmaEnv(base_url, timeout)
        self._loop = asyncio.new_event_loop()

    def __enter__(self):
        self._loop.run_until_complete(self._async_env.__aenter__())
        return self

    def __exit__(self, *args):
        self._loop.run_until_complete(self._async_env.__aexit__(*args))
        self._loop.close()

    def reset(self) -> VishwakarmaObservation:
        return self._loop.run_until_complete(self._async_env.reset())

    def step(self, action: VishwakarmaAction) -> VishwakarmaObservation:
        return self._loop.run_until_complete(self._async_env.step(action))

    def state(self) -> dict:
        return self._loop.run_until_complete(self._async_env.state())
