"""
vishwakarma_env/server/app.py
FastAPI server that exposes the Vishwakarma environment via HTTP.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .vishwakarma_environment import VishwakarmaEnvironment

app = FastAPI(
    title="Vishwakarma Factory Environment",
    description="RL environment for AI-driven factory operations. "
                "Train LLMs to manage Indian manufacturing plants.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global environment instance
factory_id = os.getenv("FACTORY_ID", "auto_components_pune")
env = VishwakarmaEnvironment(factory_id=factory_id)


# ─────────────────────────────────────────────
# Pydantic request/response models
# ─────────────────────────────────────────────

class ActionRequest(BaseModel):
    directive: str
    reroute_from: Optional[str] = None
    reroute_to: Optional[str] = None
    call_maintenance: bool = False
    order_stock_tons: float = 0.0
    order_stock_supplier: str = "primary"
    authorize_overtime_workers: int = 0
    call_contractors: int = 0
    adjust_production_rate: float = 1.0
    accept_emergency_order: bool = False
    reasoning: str = ""


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/reset")
async def reset():
    """Start a new factory episode."""
    obs = env.reset()
    return _serialize_obs(obs)


@app.post("/step")
async def step(action: ActionRequest):
    """Execute one time step with the given action."""
    from ..models import VishwakarmaAction
    act = VishwakarmaAction(**action.dict())
    obs = env.step(act)
    return _serialize_obs(obs)


@app.get("/state")
async def state():
    """Get current episode metadata."""
    return env.state_info()


@app.get("/health")
async def health():
    return {"status": "ok", "factory": factory_id}


@app.get("/factories")
async def list_factories():
    """List available factory configurations."""
    return {
        "factories": [
            "auto_components_pune",
            "pharma_packaging_hyderabad",
            "textile_mill_surat"
        ]
    }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _serialize_obs(obs) -> dict:
    """Convert observation dataclass to JSON-serializable dict."""
    return {
        "machines": [
            {
                "name": m.name,
                "online": m.online,
                "utilization": m.utilization,
                "breakdown_eta_mins": m.breakdown_eta_mins
            }
            for m in obs.machines
        ],
        "workers_present": obs.workers_present,
        "workers_total": obs.workers_total,
        "stock_tons": obs.stock_tons,
        "stock_days_remaining": obs.stock_days_remaining,
        "units_produced_today": obs.units_produced_today,
        "units_target_today": obs.units_target_today,
        "production_rate_per_hour": obs.production_rate_per_hour,
        "cost_today_INR": obs.cost_today_INR,
        "budget_today_INR": obs.budget_today_INR,
        "cumulative_cost_INR": obs.cumulative_cost_INR,
        "cumulative_budget_INR": obs.cumulative_budget_INR,
        "active_alerts": [
            {
                "crisis_type": a.crisis_type,
                "message": a.message,
                "severity": a.severity,
                "resolution_options": a.resolution_options
            }
            for a in obs.active_alerts
        ],
        "crisis_type": obs.crisis_type,
        "crisis_severity": obs.crisis_severity,
        "reward": obs.reward,
        "reward_breakdown": obs.reward_breakdown,
        "done": obs.done,
        "step": obs.step,
        "shift": obs.shift,
        "day": obs.day,
        "feedback": obs.feedback
    }


import uvicorn`ndef main():
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)

if __name__ == '__main__':
    main()

