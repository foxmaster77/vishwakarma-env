"""
vishwakarma_env/server/crisis_engine.py  v3
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional
from ..models import CrisisType, Severity, Alert


@dataclass
class CrisisEvent:
    crisis_type: CrisisType
    severity: Severity
    affected_entity: str
    message: str
    resolution_options: List[str]   # human-readable hints shown to agent
    production_impact: float        # multiplier on production: 0.35–0.90
    cost_impact_INR: int            # one-time cost when crisis fires
    duration_steps: int             # expires after this many steps if unresolved
    steps_active: int = 0
    resolved: bool = False
    metadata: dict = field(default_factory=dict)

    def tick(self):
        self.steps_active += 1

    @property
    def expired(self) -> bool:
        return self.steps_active >= self.duration_steps

    def to_alert(self) -> Alert:
        return Alert(
            crisis_type=self.crisis_type,
            message=self.message,
            severity=self.severity,
            resolution_options=self.resolution_options
        )

    # Resolution check — clear per-type logic
    def is_resolved_by(self, action) -> bool:
        ct = self.crisis_type
        if ct == CrisisType.MACHINE_BREAKDOWN:
            return action.call_maintenance or (
                bool(action.reroute_from) and bool(action.reroute_to)
            )
        elif ct == CrisisType.SUPPLY_SHOCK:
            return action.order_stock_tons > 0
        elif ct == CrisisType.DEMAND_SPIKE:
            return (action.accept_emergency_order or
                    action.authorize_overtime_workers > 0 or
                    action.directive == "decline_order")
        elif ct == CrisisType.QUALITY_FAILURE:
            return (action.call_maintenance or
                    action.adjust_production_rate < 0.9)
        elif ct == CrisisType.WORKER_CRISIS:
            return (action.call_contractors > 0 or
                    action.adjust_production_rate <= 0.85)
        return False


class CrisisEngine:
    """
    Manages crisis lifecycle: generation, ticking, resolution.
    One active crisis at a time.
    Guarantees at least 1 crisis between steps 3–5.
    ~25% chance of a new crisis each step after step 6.
    """

    def __init__(self, factory_config: dict, rng_seed=None):
        self.factory    = factory_config
        self.rng        = random.Random(rng_seed)
        self.step_count = 0

        # Guarantee at least 1 crisis early
        self.guaranteed_step   = self.rng.randint(3, 5)
        self.guaranteed_fired  = False

        self.machine_age:   dict  = {m: 0 for m in factory_config.get("machines", [])}
        self.worker_fatigue: float = 0.0

        self.active_crisis: Optional[CrisisEvent] = None

    def tick(self, state, action):
        """
        Call every step.
        Returns (new_crisis | None, status_string)
        status: "new" | "resolved" | "expired" | "active" | "none"
        """
        self.step_count += 1
        self._update_risk(action)

        # ── Handle existing crisis ──────────────────────────
        if self.active_crisis and not self.active_crisis.resolved:
            if self.active_crisis.is_resolved_by(action):
                self.active_crisis.resolved = True
                self.active_crisis = None
                return None, "resolved"

            self.active_crisis.tick()
            if self.active_crisis.expired:
                self.active_crisis = None
                return None, "expired"

            return None, "active"

        # ── Generate new crisis ─────────────────────────────
        # Guaranteed window
        if not self.guaranteed_fired and self.step_count >= self.guaranteed_step:
            self.guaranteed_fired = True
            c = self._build(state)
            self.active_crisis = c
            return c, "new"

        # Random after step 6
        if self.step_count > 6 and self.rng.random() < 0.25:
            c = self._build(state)
            self.active_crisis = c
            return c, "new"

        # Small chance before guaranteed fires
        if not self.guaranteed_fired and self.rng.random() < 0.05:
            self.guaranteed_fired = True
            c = self._build(state)
            self.active_crisis = c
            return c, "new"

        return None, "none"

    def notify_maintenance_done(self, machine_id: str):
        self.machine_age[machine_id] = 0

    # ── Crisis builders ─────────────────────────────────────

    def _build(self, state) -> CrisisEvent:
        weights = {
            CrisisType.MACHINE_BREAKDOWN: 0.25 + min(sum(self.machine_age.values()) / 200, 0.15),
            CrisisType.SUPPLY_SHOCK:      0.20 + max(0, (2.0 - getattr(state, "buffer_stock_days", 2)) * 0.08),
            CrisisType.DEMAND_SPIKE:      0.20,
            CrisisType.QUALITY_FAILURE:   0.18,
            CrisisType.WORKER_CRISIS:     0.17 + self.worker_fatigue * 0.10,
        }
        chosen = self.rng.choices(list(weights.keys()), weights=list(weights.values()), k=1)[0]
        return {
            CrisisType.MACHINE_BREAKDOWN: self._machine_breakdown,
            CrisisType.SUPPLY_SHOCK:      self._supply_shock,
            CrisisType.DEMAND_SPIKE:      self._demand_spike,
            CrisisType.QUALITY_FAILURE:   self._quality_failure,
            CrisisType.WORKER_CRISIS:     self._worker_crisis,
        }[chosen](state)

    def _sev(self):
        return self.rng.choices(
            [Severity.LOW, Severity.MEDIUM, Severity.HIGH],
            weights=[0.45, 0.38, 0.17]
        )[0]

    def _machine_breakdown(self, state) -> CrisisEvent:
        machines = self.factory.get("machines", ["cnc_mill"])
        machine  = self.rng.choice(machines)
        sev      = self._sev()
        impact   = {Severity.LOW: 0.82, Severity.MEDIUM: 0.62, Severity.HIGH: 0.42}[sev]
        cause    = self.rng.choice(["bearing failure","coolant leak","spindle overheat",
                                    "electrical fault","tool breakage","hydraulic loss"])
        return CrisisEvent(
            crisis_type=CrisisType.MACHINE_BREAKDOWN, severity=sev,
            affected_entity=machine,
            message=f"{machine.replace('_',' ').title()} breakdown — {cause}",
            resolution_options=["call_maintenance=True", "reroute_from=X reroute_to=Y"],
            production_impact=impact, cost_impact_INR=self.rng.randint(3000, 9000),
            duration_steps=self.rng.randint(3, 6), metadata={"machine_id": machine}
        )

    def _supply_shock(self, state) -> CrisisEvent:
        delay  = self.rng.choice([2, 4, 8])
        sev    = Severity.LOW if delay <= 2 else Severity.MEDIUM if delay <= 4 else Severity.HIGH
        impact = {Severity.LOW: 0.85, Severity.MEDIUM: 0.65, Severity.HIGH: 0.40}[sev]
        reason = self.rng.choice(["transport strike on NH-48","lorry breakdown near Khopoli",
                                   "mill shutdown due to power cut",
                                   "floods on Pune-Mumbai expressway",
                                   "customs clearance delay at Mumbai port"])
        return CrisisEvent(
            crisis_type=CrisisType.SUPPLY_SHOCK, severity=sev,
            affected_entity="primary_supplier",
            message=f"Primary supplier delayed {delay}hrs — {reason}",
            resolution_options=["order_stock_tons=2.0", "order_stock_supplier=backup"],
            production_impact=impact, cost_impact_INR=0,
            duration_steps=max(2, delay // 2), metadata={"delay_hours": delay}
        )

    def _demand_spike(self, state) -> CrisisEvent:
        pct   = self.rng.choice([15, 25, 40])
        sev   = Severity.LOW if pct <= 15 else Severity.MEDIUM if pct <= 25 else Severity.HIGH
        extra = int(getattr(state, "units_target", 600) * pct / 100)
        src   = self.rng.choice(["Maruti Suzuki urgent KANBAN","Tata Motors assembly shortage",
                                  "export order from Thailand plant","competitor factory shutdown"])
        return CrisisEvent(
            crisis_type=CrisisType.DEMAND_SPIKE, severity=sev,
            affected_entity="customer_order",
            message=f"Urgent order: +{extra} units (+{pct}%) — {src}",
            resolution_options=["accept_emergency_order=True", "authorize_overtime_workers=6",
                                 "directive=decline_order"],
            production_impact=1.0, cost_impact_INR=0,
            duration_steps=2,
            metadata={"extra_units": extra, "spike_pct": pct,
                      "revenue_INR": extra * self.rng.randint(200, 350)}
        )

    def _quality_failure(self, state) -> CrisisEvent:
        pct    = self.rng.choice([8, 18, 30])
        sev    = Severity.LOW if pct <= 8 else Severity.MEDIUM if pct <= 18 else Severity.HIGH
        impact = max(0.45, 1.0 - pct / 100)
        cause  = self.rng.choice(["tool wear on CNC mill","coolant contamination",
                                   "raw material batch variation","operator fatigue error",
                                   "calibration drift"])
        rework = int(getattr(state, "units_produced", 100) * pct / 100) * self.rng.randint(50, 130)
        return CrisisEvent(
            crisis_type=CrisisType.QUALITY_FAILURE, severity=sev,
            affected_entity="production_batch",
            message=f"Quality failure: {pct}% reject rate — {cause}",
            resolution_options=["call_maintenance=True", "adjust_production_rate=0.7"],
            production_impact=impact, cost_impact_INR=rework,
            duration_steps=self.rng.randint(2, 4),
            metadata={"reject_pct": pct, "rework_cost": rework}
        )

    def _worker_crisis(self, state) -> CrisisEvent:
        absent  = self.rng.choice([3, 5, 7])
        sev     = Severity.LOW if absent <= 3 else Severity.MEDIUM if absent <= 5 else Severity.HIGH
        total   = max(getattr(state, "workers_present", 26), 1)
        impact  = max(0.50, (total - absent) / total)
        reasons = {
            Severity.LOW:    ["seasonal illness", "local festival", "transport disruption"],
            Severity.MEDIUM: ["food poisoning from canteen", "wage dispute", "injury previous shift"],
            Severity.HIGH:   ["wildcat strike threat", "mass food poisoning", "union action on overtime"],
        }
        return CrisisEvent(
            crisis_type=CrisisType.WORKER_CRISIS, severity=sev,
            affected_entity="workforce",
            message=f"{absent} workers absent — {self.rng.choice(reasons[sev])}",
            resolution_options=["call_contractors=3", "adjust_production_rate=0.8"],
            production_impact=impact,
            cost_impact_INR=absent * self.rng.randint(700, 1400),
            duration_steps=self.rng.randint(2, 5),
            metadata={"absent_count": absent}
        )

    def _update_risk(self, action):
        for m in self.machine_age:
            self.machine_age[m] += 1
        if getattr(action, "call_maintenance", False):
            self.machine_age = {m: 0 for m in self.machine_age}
        if getattr(action, "authorize_overtime_workers", 0) > 0:
            self.worker_fatigue = min(1.0, self.worker_fatigue + 0.06)
        else:
            self.worker_fatigue = max(0.0, self.worker_fatigue - 0.02)
