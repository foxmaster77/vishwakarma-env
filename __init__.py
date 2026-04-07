"""
vishwakarma_env
===============
An OpenEnv-compatible RL environment for AI-driven factory operations.

Train LLMs to manage Indian manufacturing plants — responding to
machine breakdowns, supply shocks, demand spikes, quality failures,
and worker crises across a simulated factory work day.

Quick start:
    from vishwakarma_env import VishwakarmaEnv, VishwakarmaAction

    # Async
    async with VishwakarmaEnv(base_url="https://...hf.space") as env:
        obs = await env.reset()
        while not obs.done:
            action = VishwakarmaAction(directive="run_normal")
            obs = await env.step(action)
            print(f"Step {obs.step}: reward={obs.reward}, units={obs.units_produced_today}")

    # Sync
    with VishwakarmaEnv(base_url="...").sync() as env:
        obs = env.reset()
        obs = env.step(VishwakarmaAction(directive="run_normal"))
"""

from .models import (
    VishwakarmaAction,
    VishwakarmaObservation,
    VishwakarmaState,
    FactoryConfig,
    MachineStatus,
    Alert,
    CrisisType,
    Severity,
    ProductionDirective,
)
from .client import VishwakarmaEnv

__all__ = [
    "VishwakarmaEnv",
    "VishwakarmaAction",
    "VishwakarmaObservation",
    "VishwakarmaState",
    "FactoryConfig",
    "MachineStatus",
    "Alert",
    "CrisisType",
    "Severity",
    "ProductionDirective",
]

__version__ = "1.0.0"
__author__  = "vishwakarma-env contributors"
