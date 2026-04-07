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


# ── Leaderboard ───────────────────────────────────────────
_leaderboard = []   # in-memory, resets on container restart

@app.post("/episode_complete")
async def record_score(payload: dict):
    """Record a completed episode score to the leaderboard."""
    entry = {
        "name":    payload.get("name", "anonymous"),
        "score":   round(float(payload.get("score", 0)), 2),
        "factory": payload.get("factory", factory_id),
        "units":   payload.get("units", 0),
        "crises_resolved": payload.get("crises_resolved", 0),
    }
    _leaderboard.append(entry)
    _leaderboard.sort(key=lambda x: x["score"], reverse=True)
    # Keep top 50
    del _leaderboard[50:]
    return {"rank": next(i+1 for i,e in enumerate(_leaderboard) if e == entry)}


@app.get("/leaderboard")
async def get_leaderboard():
    """Top episode scores — updated live as agents play."""
    return {
        "leaderboard": _leaderboard[:10],
        "total_episodes": len(_leaderboard),
        "factory": factory_id,
    }


# ── Multi-agent endpoint ──────────────────────────────────
from .multi_agent_environment import MultiAgentVishwakarmaEnvironment, MultiAgentAction
from ..models import VishwakarmaAction as VA

_multi_env = MultiAgentVishwakarmaEnvironment()

@app.post("/multi/reset")
async def multi_reset():
    """Start a new multi-factory episode."""
    obs = _multi_env.reset()
    return _serialize_multi_obs(obs)

@app.post("/multi/step")
async def multi_step(payload: dict):
    """Step the multi-factory environment."""
    action = MultiAgentAction(
        factory_a=VA(directive=payload.get("directive_a", "run_normal"),
                     order_stock_tons=payload.get("stock_tons_a", 0),
                     call_maintenance=payload.get("maintenance_a", False),
                     call_contractors=payload.get("contractors_a", 0),
                     reasoning=payload.get("reasoning", "")),
        factory_b=VA(directive=payload.get("directive_b", "run_normal"),
                     order_stock_tons=payload.get("stock_tons_b", 0),
                     call_maintenance=payload.get("maintenance_b", False),
                     call_contractors=payload.get("contractors_b", 0),
                     reasoning=payload.get("reasoning", "")),
        stock_split_to_a=float(payload.get("stock_split_to_a", 0.5)),
        reasoning=payload.get("reasoning", ""),
    )
    obs = _multi_env.step(action)
    return _serialize_multi_obs(obs)

@app.get("/multi/state")
async def multi_state():
    return _multi_env.state_info()

def _serialize_multi_obs(obs) -> dict:
    return {
        "factory_a":              _serialize_obs(obs.factory_a) if obs.factory_a else {},
        "factory_b":              _serialize_obs(obs.factory_b) if obs.factory_b else {},
        "supplier_stock_tons":    obs.supplier_stock_tons,
        "supplier_under_shock":   obs.supplier_under_shock,
        "combined_reward":        obs.combined_reward,
        "combined_units_produced": obs.combined_units_produced,
        "combined_units_target":  obs.combined_units_target,
        "step":                   obs.step,
        "done":                   obs.done,
        "feedback":               obs.feedback,
    }


# ── Leaderboard ──────────────────────────────────────────────

_leaderboard: list = []   # in-memory, resets on server restart

@app.post("/episode_complete")
async def record_episode(
    team_name: str,
    total_reward: float,
    units_produced: int,
    units_target: int,
    crises_resolved: int,
    crises_total: int,
    factory_id: str = "auto_components_pune"
):
    """Record a completed episode score. Called by clients after an episode."""
    global _leaderboard
    entry = {
        "team": team_name[:30],
        "reward": round(total_reward, 2),
        "units": f"{units_produced}/{units_target}",
        "crises": f"{crises_resolved}/{crises_total}",
        "factory": factory_id,
        "pct": round(units_produced / max(units_target, 1) * 100, 1),
    }
    _leaderboard.append(entry)
    # Keep top 20 by reward
    _leaderboard = sorted(_leaderboard, key=lambda x: x["reward"], reverse=True)[:20]
    return {"rank": _leaderboard.index(entry) + 1, "entry": entry}


@app.get("/leaderboard")
async def get_leaderboard():
    """Top 20 episode scores across all teams testing this environment."""
    return {
        "leaderboard": _leaderboard,
        "total_episodes_recorded": len(_leaderboard),
        "environment": "vishwakarma-env",
        "note": "Top 20 by total episode reward. Submit your score via POST /episode_complete"
    }


# ── Multi-agent ───────────────────────────────────────────────

_multi_env = None

@app.post("/multi/reset")
async def multi_reset():
    """Start a new multi-agent episode (two factories, shared supplier)."""
    global _multi_env
    from .multi_agent_environment import MultiAgentVishwakarmaEnvironment
    _multi_env = MultiAgentVishwakarmaEnvironment()
    obs_a, obs_b = _multi_env.reset()
    return {
        "factory_a": _serialize_obs(obs_a.base),
        "factory_b": _serialize_obs(obs_b.base),
        "shared_supplier_available_tons": obs_a.shared_supplier_available_tons,
        "mode": "multi_agent"
    }


@app.post("/multi/step")
async def multi_step(action_a: ActionRequest, action_b: ActionRequest):
    """Step both agents simultaneously in the multi-agent environment."""
    global _multi_env
    if _multi_env is None:
        raise HTTPException(status_code=400, detail="Call /multi/reset first")
    from ..models import VishwakarmaAction
    act_a = VishwakarmaAction(**action_a.dict())
    act_b = VishwakarmaAction(**action_b.dict())
    obs_a, obs_b = _multi_env.step(act_a, act_b)
    return {
        "factory_a": _serialize_obs(obs_a.base),
        "factory_b": _serialize_obs(obs_b.base),
        "shared_supplier_available_tons": obs_a.shared_supplier_available_tons,
        "shared_supplier_used_tons": obs_a.shared_supplier_used_tons,
        "other_a_stock_days": obs_a.other_factory_stock_days,
        "other_b_stock_days": obs_b.other_factory_stock_days,
    }


@app.get("/multi/state")
async def multi_state():
    """Multi-agent episode state."""
    if _multi_env is None:
        return {"status": "not_started"}
    return _multi_env.state_info()


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
