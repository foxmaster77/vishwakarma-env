"""
vishwakarma_env/server/crisis_engine.py

Generates and manages industrial crises during factory operation.
5 crisis types: machine breakdown, supply shock, demand spike,
quality failure, worker crisis.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from ..models import CrisisType, Severity, Alert


# ─────────────────────────────────────────────
# Crisis event dataclass
# ─────────────────────────────────────────────

@dataclass
class CrisisEvent:
    crisis_type: CrisisType
    severity: Severity
    affected_entity: str          # machine name, supplier name, etc.
    message: str
    resolution_options: List[str]
    auto_resolves_in_steps: int   # 0 = must be resolved by agent
    production_impact: float      # multiplier on production rate (0–1)
    cost_impact_INR: int          # immediate cost hit
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Crisis Engine
# ─────────────────────────────────────────────

class CrisisEngine:
    """
    Probabilistically generates crises each step.
    Crisis probability increases with:
    - Machine age (steps since last maintenance)
    - Worker fatigue (consecutive overtime)
    - Low stock levels
    - Single supplier dependency
    """

    def __init__(self, factory_config: dict, rng_seed: Optional[int] = None):
        self.factory = factory_config
        self.rng = random.Random(rng_seed)
        self.step_count = 0
        self.machine_age: dict = {}          # steps since last maintenance
        self.worker_fatigue: float = 0.0     # 0–1, increases with overtime
        self.consecutive_overtime: int = 0

    def tick(self, factory_state) -> Optional[CrisisEvent]:
        """
        Called every step. Returns a CrisisEvent or None.
        """
        self.step_count += 1
        self._update_risk_factors(factory_state)

        # Roll for each crisis type
        crisis_rolls = [
            self._roll_machine_breakdown(factory_state),
            self._roll_supply_shock(factory_state),
            self._roll_demand_spike(factory_state),
            self._roll_quality_failure(factory_state),
            self._roll_worker_crisis(factory_state),
        ]

        # Filter out None, pick the first one that triggers
        triggered = [c for c in crisis_rolls if c is not None]
        if not triggered:
            return None

        # If multiple crises roll simultaneously, pick the most severe
        triggered.sort(key=lambda c: {"low": 0, "medium": 1, "high": 2}[c.severity])
        return triggered[-1]

    # ─────────────────────────────────────────
    # Crisis type 1: Machine Breakdown
    # ─────────────────────────────────────────

    def _roll_machine_breakdown(self, state) -> Optional[CrisisEvent]:
        online_machines = [
            m for m in self.factory["machines"]
            if state.machine_online.get(m, True)
        ]
        if not online_machines:
            return None

        for machine_id in online_machines:
            base_prob = self._get_machine_breakdown_prob(machine_id)
            age_factor = min(self.machine_age.get(machine_id, 0) / 40, 2.0)
            prob = base_prob * (1 + age_factor)

            if self.rng.random() < prob:
                severity = self._classify_severity(prob, thresholds=(0.05, 0.12))
                machine_name = machine_id.replace("_", " ").title()

                return CrisisEvent(
                    crisis_type=CrisisType.MACHINE_BREAKDOWN,
                    severity=severity,
                    affected_entity=machine_id,
                    message=f"{machine_name} has broken down — {self._breakdown_cause()}",
                    resolution_options=[
                        "call_maintenance",
                        "reroute_jobs",
                        "run_degraded"
                    ],
                    auto_resolves_in_steps=0,
                    production_impact=0.75 if severity == Severity.LOW else
                                      0.50 if severity == Severity.MEDIUM else 0.30,
                    cost_impact_INR=self.rng.randint(2000, 8000),
                    metadata={
                        "repair_time_steps": 1 if severity == Severity.LOW else
                                             3 if severity == Severity.MEDIUM else 5,
                        "machine_id": machine_id
                    }
                )
        return None

    def _get_machine_breakdown_prob(self, machine_id: str) -> float:
        machine_db = {
            "cnc_mill": 0.05, "lathe": 0.04,
            "welding_station": 0.06, "press_brake": 0.03,
            "qc_station": 0.01, "painting_booth": 0.04,
            "drill_press": 0.03, "grinding_machine": 0.05
        }
        return machine_db.get(machine_id, 0.04)

    def _breakdown_cause(self) -> str:
        causes = [
            "bearing failure", "coolant leak", "spindle overheating",
            "hydraulic pressure loss", "electrical fault",
            "tool breakage", "belt snap", "sensor malfunction"
        ]
        return self.rng.choice(causes)

    # ─────────────────────────────────────────
    # Crisis type 2: Supply Shock
    # ─────────────────────────────────────────

    def _roll_supply_shock(self, state) -> Optional[CrisisEvent]:
        # Higher probability when stock is low
        stock_risk = max(0, 1 - (state.stock_tons / 5.0))
        base_prob = 0.06
        prob = base_prob + stock_risk * 0.10

        if self.rng.random() < prob:
            severity = self._classify_severity(prob, thresholds=(0.08, 0.14))
            delay_hours = self.rng.choice([2, 4, 8])
            cost_premium = self.rng.randint(8000, 25000)

            return CrisisEvent(
                crisis_type=CrisisType.SUPPLY_SHOCK,
                severity=severity,
                affected_entity="primary_supplier",
                message=f"Primary steel supplier delayed by {delay_hours} hours — "
                        f"{self._supply_shock_reason()}",
                resolution_options=[
                    "use_buffer_stock",
                    "emergency_order_backup",
                    "reduce_production_rate"
                ],
                auto_resolves_in_steps=delay_hours // 2,
                production_impact=0.85 if severity == Severity.LOW else
                                  0.60 if severity == Severity.MEDIUM else 0.30,
                cost_impact_INR=cost_premium,
                metadata={
                    "delay_hours": delay_hours,
                    "backup_cost_premium_INR": cost_premium
                }
            )
        return None

    def _supply_shock_reason(self) -> str:
        reasons = [
            "transport strike on NH-48",
            "lorry breakdown near Khopoli",
            "sudden price renegotiation",
            "mill shutdown due to power cut",
            "floods on Pune-Mumbai expressway",
            "customs clearance delay at Mumbai port"
        ]
        return self.rng.choice(reasons)

    # ─────────────────────────────────────────
    # Crisis type 3: Demand Spike
    # ─────────────────────────────────────────

    def _roll_demand_spike(self, state) -> Optional[CrisisEvent]:
        # Demand spike is an opportunity, not just a problem
        prob = 0.04
        if self.rng.random() < prob:
            spike_percent = self.rng.choice([10, 25, 50])
            extra_units = int(state.units_target * (spike_percent / 100))
            extra_revenue = extra_units * self.rng.randint(180, 320)

            severity = (Severity.LOW if spike_percent <= 10
                        else Severity.MEDIUM if spike_percent <= 25
                        else Severity.HIGH)

            return CrisisEvent(
                crisis_type=CrisisType.DEMAND_SPIKE,
                severity=severity,
                affected_entity="customer_order",
                message=f"Urgent order received: {extra_units} additional units "
                        f"(+{spike_percent}%) — {self._demand_spike_source()}",
                resolution_options=[
                    "accept_order_overtime",
                    "partial_fulfil",
                    "decline_order"
                ],
                auto_resolves_in_steps=1,
                production_impact=1.0,
                cost_impact_INR=0,
                metadata={
                    "extra_units": extra_units,
                    "potential_revenue_INR": extra_revenue,
                    "spike_percent": spike_percent
                }
            )
        return None

    def _demand_spike_source(self) -> str:
        sources = [
            "Maruti Suzuki urgent KANBAN request",
            "Tata Motors assembly line shortage",
            "export order from Thailand plant",
            "competitor factory shut down",
            "government infra project urgent supply"
        ]
        return self.rng.choice(sources)

    # ─────────────────────────────────────────
    # Crisis type 4: Quality Failure
    # ─────────────────────────────────────────

    def _roll_quality_failure(self, state) -> Optional[CrisisEvent]:
        # Higher risk if QC station is bypassed or overloaded
        qc_online = state.machine_online.get("qc_station", True)
        base_prob = 0.03 if qc_online else 0.12

        if self.rng.random() < base_prob:
            reject_rate = self.rng.choice([5, 15, 30])
            rejected_units = int(state.units_produced * (reject_rate / 100))
            rework_cost = rejected_units * self.rng.randint(45, 120)

            severity = (Severity.LOW if reject_rate <= 5
                        else Severity.MEDIUM if reject_rate <= 15
                        else Severity.HIGH)

            return CrisisEvent(
                crisis_type=CrisisType.QUALITY_FAILURE,
                severity=severity,
                affected_entity="production_batch",
                message=f"Quality inspection failed — {reject_rate}% reject rate "
                        f"({rejected_units} units). Cause: {self._quality_failure_cause()}",
                resolution_options=[
                    "rework_batch",
                    "scrap_and_restart",
                    "ship_and_flag_customer"
                ],
                auto_resolves_in_steps=0,
                production_impact=1.0,
                cost_impact_INR=rework_cost,
                metadata={
                    "reject_rate": reject_rate,
                    "rejected_units": rejected_units,
                    "rework_cost_INR": rework_cost
                }
            )
        return None

    def _quality_failure_cause(self) -> str:
        causes = [
            "tool wear on CNC mill #2",
            "coolant contamination",
            "raw material batch variation",
            "operator fatigue error",
            "machine calibration drift",
            "welding parameter deviation"
        ]
        return self.rng.choice(causes)

    # ─────────────────────────────────────────
    # Crisis type 5: Worker Crisis
    # ─────────────────────────────────────────

    def _roll_worker_crisis(self, state) -> Optional[CrisisEvent]:
        # Fatigue increases risk
        base_prob = 0.04 + self.worker_fatigue * 0.08

        if self.rng.random() < base_prob:
            absent_count = self.rng.choice([2, 4, 6])
            severity = (Severity.LOW if absent_count <= 2
                        else Severity.MEDIUM if absent_count <= 4
                        else Severity.HIGH)

            contractor_cost = absent_count * self.rng.randint(800, 1500)
            reason = self._worker_crisis_reason(severity)

            return CrisisEvent(
                crisis_type=CrisisType.WORKER_CRISIS,
                severity=severity,
                affected_entity="workforce",
                message=f"{absent_count} workers absent — {reason}",
                resolution_options=[
                    "call_contractors",
                    "redistribute_tasks",
                    "reduce_production_rate"
                ],
                auto_resolves_in_steps=0 if severity == Severity.HIGH else 2,
                production_impact=max(0.4,
                    (state.workers_present - absent_count) / state.workers_present),
                cost_impact_INR=contractor_cost,
                metadata={
                    "absent_count": absent_count,
                    "contractor_cost_INR": contractor_cost
                }
            )
        return None

    def _worker_crisis_reason(self, severity: Severity) -> str:
        reasons = {
            Severity.LOW: [
                "seasonal illness", "personal emergency",
                "local festival", "transport disruption"
            ],
            Severity.MEDIUM: [
                "food poisoning from canteen",
                "wage dispute",
                "injury in previous shift"
            ],
            Severity.HIGH: [
                "wildcat strike threat over safety concerns",
                "mass food poisoning",
                "union action over overtime policy"
            ]
        }
        return self.rng.choice(reasons[severity])

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _classify_severity(self, prob: float,
                            thresholds: Tuple[float, float]) -> Severity:
        if prob < thresholds[0]:
            return Severity.LOW
        elif prob < thresholds[1]:
            return Severity.MEDIUM
        else:
            return Severity.HIGH

    def _update_risk_factors(self, state):
        """Update machine age and worker fatigue each step."""
        for machine_id in self.factory["machines"]:
            if state.machine_online.get(machine_id, True):
                self.machine_age[machine_id] = \
                    self.machine_age.get(machine_id, 0) + 1
            else:
                self.machine_age[machine_id] = 0  # reset after repair

        # Fatigue increases with overtime, decreases otherwise
        if state.overtime_authorized > 0:
            self.consecutive_overtime += 1
            self.worker_fatigue = min(1.0, self.worker_fatigue + 0.05)
        else:
            self.consecutive_overtime = 0
            self.worker_fatigue = max(0.0, self.worker_fatigue - 0.02)

    def notify_maintenance_done(self, machine_id: str):
        """Reset machine age after maintenance."""
        self.machine_age[machine_id] = 0
