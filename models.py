"""
vishwakarma_env/models.py
Core data models for the Vishwakarma Factory Environment.
Action, Observation, and State definitions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class ProductionDirective(str, Enum):
    RUN_NORMAL        = "run_normal"
    REROUTE_JOBS      = "reroute_jobs"
    CALL_MAINTENANCE  = "call_maintenance"
    ORDER_STOCK       = "order_stock"
    AUTHORIZE_OVERTIME = "authorize_overtime"
    CALL_CONTRACTOR   = "call_contractor"
    ACCEPT_ORDER      = "accept_order"
    DECLINE_ORDER     = "decline_order"
    ADJUST_RATE       = "adjust_rate"
    FINALIZE_SHIFT    = "finalize_shift"


class CrisisType(str, Enum):
    MACHINE_BREAKDOWN = "machine_breakdown"
    SUPPLY_SHOCK      = "supply_shock"
    DEMAND_SPIKE      = "demand_spike"
    QUALITY_FAILURE   = "quality_failure"
    WORKER_CRISIS     = "worker_crisis"
    NONE              = "none"


class Severity(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ─────────────────────────────────────────────
# Action — what the agent sends each step
# ─────────────────────────────────────────────

@dataclass
class VishwakarmaAction:
    """
    The LLM agent's decision for this time step.
    Not all fields are required every step — depends on the crisis.
    """
    directive: str                              # ProductionDirective value
    reroute_from: Optional[str] = None          # machine name to reroute FROM
    reroute_to: Optional[str] = None            # machine name to reroute TO
    call_maintenance: bool = False              # dispatch maintenance team
    order_stock_tons: float = 0.0              # emergency stock order qty
    order_stock_supplier: str = "primary"      # "primary" or "backup"
    authorize_overtime_workers: int = 0        # how many workers for overtime
    call_contractors: int = 0                  # external contractors to hire
    adjust_production_rate: float = 1.0        # 0.5 = half speed, 1.0 = full
    accept_emergency_order: bool = False        # accept spike demand order
    reasoning: str = ""                        # CoT — agent explains its decision


# ─────────────────────────────────────────────
# Observation — what the environment returns
# ─────────────────────────────────────────────

@dataclass
class MachineStatus:
    name: str
    online: bool
    utilization: float          # 0.0 – 1.0
    breakdown_eta_mins: int = 0 # 0 = not broken, >0 = repair ETA


@dataclass
class Alert:
    crisis_type: str
    message: str
    severity: str
    resolution_options: List[str]


@dataclass
class VishwakarmaObservation:
    """
    Full state the agent sees after each step.
    """
    # Factory status
    machines: List[MachineStatus] = field(default_factory=list)
    workers_present: int = 0
    workers_total: int = 0
    stock_tons: float = 0.0
    stock_days_remaining: float = 0.0

    # Production
    units_produced_today: int = 0
    units_target_today: int = 0
    production_rate_per_hour: float = 0.0

    # Financials
    cost_today_INR: int = 0
    budget_today_INR: int = 0
    cumulative_cost_INR: int = 0
    cumulative_budget_INR: int = 0

    # Crisis
    active_alerts: List[Alert] = field(default_factory=list)
    crisis_type: str = CrisisType.NONE
    crisis_severity: str = Severity.LOW

    # Step result
    reward: float = 0.0
    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    done: bool = False
    step: int = 0
    shift: int = 1
    day: int = 1
    feedback: str = ""          # plain English explanation of what happened


# ─────────────────────────────────────────────
# State — full internal episode state (server-side)
# ─────────────────────────────────────────────

@dataclass
class FactoryConfig:
    name: str
    industry: str
    floor_width_m: float
    floor_length_m: float
    total_machines: int
    total_workers: int
    shifts_per_day: int
    target_units_per_day: int
    budget_per_day_INR: int
    machines: List[dict]        # machine specs from JSON
    suppliers: List[dict]       # supplier profiles


@dataclass
class VishwakarmaState:
    """
    Full mutable factory state — lives on the server.
    Not exposed directly to the agent; summarized into Observation.
    """
    episode_id: str
    factory_config: FactoryConfig

    # Time
    day: int = 1
    shift: int = 1
    step: int = 0
    steps_per_shift: int = 8
    total_steps: int = 16        # 2 shifts × 8 steps = 1 full factory day

    # Machine health
    machine_health: Dict[str, float] = field(default_factory=dict)  # 0–1
    machine_online: Dict[str, bool] = field(default_factory=dict)
    maintenance_eta: Dict[str, int] = field(default_factory=dict)   # steps remaining

    # Workforce
    workers_present: int = 0
    contractors_hired: int = 0
    overtime_authorized: int = 0

    # Stock
    stock_tons: float = 0.0
    daily_consumption_tons: float = 0.0
    buffer_stock_days: float = 0.0

    # Production
    units_produced: int = 0
    units_target: int = 0
    production_rate: float = 0.0

    # Financials
    cost_today: int = 0
    budget_today: int = 0
    cumulative_cost: int = 0
    cumulative_budget: int = 0

    # Crisis tracking
    active_crisis: CrisisType = CrisisType.NONE
    crisis_severity: Severity = Severity.LOW
    crisis_step: int = 0
    crises_resolved: int = 0
    crisis_history: List[dict] = field(default_factory=list)

    # Supplier relationships
    supplier_reliability: Dict[str, float] = field(default_factory=dict)
    supplier_diversity: int = 1

    # Long-term health scores
    safety_incidents: int = 0
    quality_reject_rate: float = 0.0
    cumulative_reward: float = 0.0
