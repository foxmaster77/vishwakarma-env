"""
vishwakarma_env/server/vishwakarma_environment.py

Main environment class. Orchestrates episodes, applies actions,
triggers crises, computes rewards, and returns observations.
"""

import uuid
import json
import os
from typing import Optional

from ..models import (
    VishwakarmaAction, VishwakarmaObservation, VishwakarmaState,
    FactoryConfig, MachineStatus, Alert, CrisisType, Severity
)
from .crisis_engine import CrisisEngine
from .production_simulator import ProductionSimulator
from .reward_engine import RewardEngine


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load_json(filename: str) -> dict:
    with open(os.path.join(DATA_DIR, filename)) as f:
        return json.load(f)


class VishwakarmaEnvironment:
    """
    The Vishwakarma Factory Environment.

    An LLM agent operates a running Indian manufacturing factory,
    responding to machine breakdowns, supply shocks, demand spikes,
    quality failures, and worker crises over a simulated work day.
    """

    STEPS_PER_SHIFT = 8
    SHIFTS_PER_DAY  = 2
    TOTAL_STEPS     = STEPS_PER_SHIFT * SHIFTS_PER_DAY   # 16 steps = 1 factory day

    def __init__(self, factory_id: str = "auto_components_pune",
                 seed: Optional[int] = None):
        # Load data
        self.factories_db  = {f["id"]: f for f in _load_json("factories.json")["factories"]}
        self.machines_db   = {m["id"]: m for m in _load_json("machines.json")["machines"]}
        self.suppliers_db  = {s["id"]: s for s in _load_json("suppliers.json")["suppliers"]}

        self.factory_id = factory_id
        self.seed = seed

        # Sub-systems
        self.crisis_engine  = None
        self.simulator      = ProductionSimulator(self.machines_db)
        self.reward_engine  = RewardEngine()

        # Episode state
        self.state: Optional[VishwakarmaState] = None
        self._pending_crisis = None

    # ─────────────────────────────────────────
    # OpenEnv API: reset()
    # ─────────────────────────────────────────

    def reset(self) -> VishwakarmaObservation:
        """Start a new episode. Returns initial observation."""
        factory_cfg = self.factories_db[self.factory_id]

        config = FactoryConfig(
            name=factory_cfg["name"],
            industry=factory_cfg["industry"],
            floor_width_m=factory_cfg["floor_width_m"],
            floor_length_m=factory_cfg["floor_length_m"],
            total_machines=factory_cfg["total_machines"],
            total_workers=factory_cfg["total_workers"],
            shifts_per_day=factory_cfg["shifts_per_day"],
            target_units_per_day=factory_cfg["target_units_per_day"],
            budget_per_day_INR=factory_cfg["budget_per_day_INR"],
            machines=factory_cfg["machines"],
            suppliers=[
                self.suppliers_db[factory_cfg["primary_supplier"]],
                self.suppliers_db[factory_cfg["backup_supplier"]],
            ]
        )

        # Initialize state
        self.state = VishwakarmaState(
            episode_id=str(uuid.uuid4())[:8],
            factory_config=config,
            workers_present=config.total_workers,
            stock_tons=factory_cfg["initial_stock_tons"],
            daily_consumption_tons=factory_cfg["daily_stock_consumption_tons"],
            units_target=config.target_units_per_day,
            budget_today=config.budget_per_day_INR,
            cumulative_budget=config.budget_per_day_INR,
        )

        # Initialize all machines online
        for machine_id in config.machines:
            self.state.machine_online[machine_id] = True
            self.state.machine_health[machine_id] = 1.0
            self.state.maintenance_eta[machine_id] = 0

        # Initialize suppliers
        for s in config.suppliers:
            self.state.supplier_reliability[s["id"]] = s["reliability"]
        self.state.supplier_diversity = len(config.suppliers)

        # Init crisis engine
        self.crisis_engine = CrisisEngine(
            factory_config={"machines": config.machines},
            rng_seed=self.seed
        )
        self.crisis_engine.machine_age = {m: 0 for m in config.machines}
        self.crisis_engine.worker_fatigue = 0.0

        self._pending_crisis = None

        return self._build_observation(
            reward=0.0,
            reward_breakdown={},
            feedback=f"Factory '{config.name}' ready. "
                     f"Target: {config.target_units_per_day} units today. "
                     f"Budget: ₹{config.budget_per_day_INR:,}. "
                     f"Workers: {config.total_workers}. "
                     f"All {config.total_machines} machines online. Good luck."
        )

    # ─────────────────────────────────────────
    # OpenEnv API: step(action)
    # ─────────────────────────────────────────

    def step(self, action: VishwakarmaAction) -> VishwakarmaObservation:
        """Execute one time step. Returns updated observation with reward."""
        if self.state is None:
            raise RuntimeError("Call reset() before step()")

        s = self.state
        s.step += 1

        feedback_lines = []
        crisis_occurred = self._pending_crisis is not None
        crisis_resolved = False
        response_delay_steps = 0
        production_loss_ratio = 0.0
        rerouted_successfully = False
        safety_incidents_this_step = 0
        production_multiplier = 1.0

        # ── 1. Apply action ──────────────────
        directive = action.directive
        step_cost = self._base_step_cost()

        # Run normal
        if directive == "run_normal":
            feedback_lines.append("Running normal operations.")

        # Reroute jobs
        if directive == "reroute_jobs" and action.reroute_from and action.reroute_to:
            mult = self.simulator.compute_reroute_impact(
                action.reroute_from, action.reroute_to,
                self._active_machine_counts()
            )
            production_multiplier = mult
            rerouted_successfully = mult >= 0.85
            feedback_lines.append(
                f"Rerouted {action.reroute_from} → {action.reroute_to}. "
                f"Production at {mult:.0%} capacity."
            )

        # Call maintenance
        if action.call_maintenance:
            for machine_id, online in s.machine_online.items():
                if not online:
                    eta = s.maintenance_eta.get(machine_id, 3)
                    s.maintenance_eta[machine_id] = max(0, eta - 2)
                    step_cost += 4500
                    feedback_lines.append(
                        f"Maintenance dispatched to {machine_id}. "
                        f"ETA: {s.maintenance_eta[machine_id]} steps."
                    )
                    if s.maintenance_eta[machine_id] == 0:
                        s.machine_online[machine_id] = True
                        s.machine_health[machine_id] = 0.85
                        crisis_resolved = True
                        self.crisis_engine.notify_maintenance_done(machine_id)
                        feedback_lines.append(f"✓ {machine_id} back online.")

        # Order stock
        if action.order_stock_tons > 0:
            supplier_key = (
                "backup_steel_mumbai"
                if action.order_stock_supplier == "backup"
                else "primary_steel_pune"
            )
            supplier = self.suppliers_db.get(supplier_key, {})
            cost_per_ton = supplier.get("cost_per_ton_INR", 52000)
            stock_cost = int(action.order_stock_tons * cost_per_ton)
            s.stock_tons += action.order_stock_tons
            step_cost += stock_cost
            feedback_lines.append(
                f"Ordered {action.order_stock_tons}t from "
                f"{supplier.get('name','supplier')}. "
                f"Cost: ₹{stock_cost:,}. Stock: {s.stock_tons:.1f}t."
            )
            if self._pending_crisis and \
               self._pending_crisis.crisis_type == CrisisType.SUPPLY_SHOCK:
                crisis_resolved = True

        # Authorize overtime
        if action.authorize_overtime_workers > 0:
            overtime_cost = action.authorize_overtime_workers * 850
            step_cost += overtime_cost
            s.overtime_authorized = action.authorize_overtime_workers
            production_multiplier = min(production_multiplier * 1.15, 1.2)
            feedback_lines.append(
                f"Overtime authorized for {action.authorize_overtime_workers} workers. "
                f"Cost: ₹{overtime_cost:,}. Production boost: +15%."
            )

        # Call contractors
        if action.call_contractors > 0:
            contractor_cost = action.call_contractors * 1200
            step_cost += contractor_cost
            s.workers_present = min(
                s.workers_present + action.call_contractors,
                s.factory_config.total_workers + 6
            )
            feedback_lines.append(
                f"Hired {action.call_contractors} contractors. "
                f"Cost: ₹{contractor_cost:,}. "
                f"Floor workforce: {s.workers_present}."
            )
            if self._pending_crisis and \
               self._pending_crisis.crisis_type == CrisisType.WORKER_CRISIS:
                crisis_resolved = True

        # Adjust production rate
        if action.adjust_production_rate != 1.0:
            production_multiplier *= action.adjust_production_rate
            feedback_lines.append(
                f"Production rate adjusted to {action.adjust_production_rate:.0%}."
            )

        # ── 2. Tick maintenance ETAs ─────────
        for machine_id in list(s.maintenance_eta.keys()):
            if s.maintenance_eta[machine_id] > 0:
                s.maintenance_eta[machine_id] -= 1
                if s.maintenance_eta[machine_id] == 0:
                    s.machine_online[machine_id] = True
                    s.machine_health[machine_id] = 0.9
                    self.crisis_engine.notify_maintenance_done(machine_id)
                    feedback_lines.append(f"✓ {machine_id} repairs complete — back online.")

        # ── 3. Simulate production this step ─
        steps_per_shift = self.STEPS_PER_SHIFT
        target_this_step = s.units_target // (self.TOTAL_STEPS)

        sim = self.simulator.simulate(
            active_machines=self._active_machine_counts(),
            workers_available=s.workers_present,
            target_units=target_this_step * steps_per_shift,
            production_rate_multiplier=production_multiplier
        )

        units_this_step = int(sim.throughput_per_hour *
                              (8 / steps_per_shift) *
                              production_multiplier)
        s.units_produced += units_this_step

        # Stock consumption
        stock_consumed = s.daily_consumption_tons / self.TOTAL_STEPS
        s.stock_tons = max(0, s.stock_tons - stock_consumed)
        s.buffer_stock_days = (
            s.stock_tons / s.daily_consumption_tons
            if s.daily_consumption_tons > 0 else 0
        )

        if s.stock_tons <= 0:
            units_this_step = 0
            s.units_produced = max(0, s.units_produced - units_this_step)
            feedback_lines.append("⚠ No raw material stock — production halted this step!")

        # ── 4. Safety check ──────────────────
        safety = self.simulator.safety_check(
            active_machines=self._active_machine_counts(),
            workers_present=s.workers_present
        )
        if not safety["passed"] and s.step == 1:
            # Only count safety violations on first step (persistent structural issues)
            # Step-by-step violations are warnings, not incidents
            safety_incidents_this_step = 0
            for v in safety["violations"]:
                feedback_lines.append(f"⚠ SAFETY WARNING: {v}")
        elif not safety["passed"]:
            safety_incidents_this_step = 0
            for v in safety["violations"]:
                feedback_lines.append(f"⚠ SAFETY WARNING: {v}")

        # ── 5. Update financials ─────────────
        machine_running_cost = self._compute_machine_cost()
        worker_cost = s.workers_present * 280   # ₹280/worker/step (approx)
        total_step_cost = step_cost + machine_running_cost + worker_cost

        s.cost_today += total_step_cost
        s.cumulative_cost += total_step_cost

        budget_this_step = s.budget_today // self.TOTAL_STEPS
        production_loss_ratio = max(
            0, 1 - (units_this_step / max(target_this_step, 1))
        )

        # ── 6. Trigger new crisis ─────────────
        new_crisis = self.crisis_engine.tick(s)
        if new_crisis and self._pending_crisis is None:
            self._pending_crisis = new_crisis
            s.active_crisis = new_crisis.crisis_type
            s.crisis_severity = new_crisis.severity
            feedback_lines.append(
                f"\n🚨 CRISIS: {new_crisis.message}"
            )
            feedback_lines.append(
                f"Options: {', '.join(new_crisis.resolution_options)}"
            )

        # Clear resolved crisis
        if crisis_resolved and self._pending_crisis:
            s.crises_resolved += 1
            s.crisis_history.append({
                "type": self._pending_crisis.crisis_type,
                "severity": self._pending_crisis.severity,
                "resolved_at_step": s.step,
                "resolved": True
            })
            self._pending_crisis = None
            s.active_crisis = CrisisType.NONE

        # ── 7. Compute reward ─────────────────
        reward_result = self.reward_engine.compute(
            units_produced=units_this_step,
            units_target=target_this_step,
            cost_this_step=total_step_cost,
            budget_this_step=budget_this_step,
            crisis_occurred=crisis_occurred,
            crisis_resolved=crisis_resolved,
            response_delay_steps=response_delay_steps,
            production_loss_ratio=production_loss_ratio,
            safety_incidents=safety_incidents_this_step,
            safety_passed=safety["passed"],
            stock_buffer_days=s.buffer_stock_days,
            supplier_diversity=s.supplier_diversity,
            rerouted_successfully=rerouted_successfully
        )

        s.cumulative_reward += reward_result.total

        # ── 8. Check episode done ─────────────
        done = s.step >= self.TOTAL_STEPS
        if done:
            episode_bonus = self.reward_engine.episode_final_reward(
                total_units=s.units_produced,
                target_units=s.units_target,
                total_cost=s.cumulative_cost,
                total_budget=s.cumulative_budget,
                total_crises=len(s.crisis_history),
                crises_resolved=s.crises_resolved,
                safety_incidents=s.safety_incidents,
                avg_buffer_days=s.buffer_stock_days
            )
            reward_result.total += episode_bonus
            s.cumulative_reward += episode_bonus
            feedback_lines.append(
                f"\n{'='*50}"
                f"\nEPISODE COMPLETE"
                f"\n  Units: {s.units_produced}/{s.units_target}"
                f"\n  Cost:  ₹{s.cumulative_cost:,} / ₹{s.cumulative_budget:,}"
                f"\n  Crises resolved: {s.crises_resolved}/{len(s.crisis_history)}"
                f"\n  Safety incidents: {s.safety_incidents}"
                f"\n  Episode bonus: {episode_bonus:+.2f}"
                f"\n  Total reward: {s.cumulative_reward:.4f}"
                f"\n{'='*50}"
            )

        # Add sim notes
        for note in sim.notes:
            feedback_lines.append(note)
        feedback_lines.append(
            f"Step {s.step}/{self.TOTAL_STEPS} | "
            f"Units today: {s.units_produced}/{s.units_target} | "
            f"Cost: ₹{s.cost_today:,}"
        )

        return self._build_observation(
            reward=reward_result.total,
            reward_breakdown={
                "production": reward_result.production,
                "cost": reward_result.cost,
                "crisis": reward_result.crisis_response,
                "safety": reward_result.safety,
                "long_term": reward_result.long_term,
            },
            feedback="\n".join(feedback_lines),
            done=done
        )

    # ─────────────────────────────────────────
    # OpenEnv API: state()
    # ─────────────────────────────────────────

    def state_info(self) -> dict:
        """Returns episode metadata."""
        if not self.state:
            return {}
        s = self.state
        return {
            "episode_id": s.episode_id,
            "step": s.step,
            "total_steps": self.TOTAL_STEPS,
            "day": s.day,
            "shift": s.shift,
            "cumulative_reward": s.cumulative_reward,
            "units_produced": s.units_produced,
            "units_target": s.units_target,
            "crises_total": len(s.crisis_history),
            "crises_resolved": s.crises_resolved,
            "safety_incidents": s.safety_incidents,
        }

    # ─────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────

    def _build_observation(self, reward: float, reward_breakdown: dict,
                            feedback: str, done: bool = False) -> VishwakarmaObservation:
        s = self.state
        machines = [
            MachineStatus(
                name=m,
                online=s.machine_online.get(m, True),
                utilization=0.85 if s.machine_online.get(m, True) else 0.0,
                breakdown_eta_mins=s.maintenance_eta.get(m, 0) * 30
            )
            for m in set(s.factory_config.machines)
        ]

        alerts = []
        if self._pending_crisis:
            c = self._pending_crisis
            alerts.append(Alert(
                crisis_type=c.crisis_type,
                message=c.message,
                severity=c.severity,
                resolution_options=c.resolution_options
            ))

        return VishwakarmaObservation(
            machines=machines,
            workers_present=s.workers_present,
            workers_total=s.factory_config.total_workers,
            stock_tons=round(s.stock_tons, 2),
            stock_days_remaining=round(s.buffer_stock_days, 1),
            units_produced_today=s.units_produced,
            units_target_today=s.units_target,
            production_rate_per_hour=round(
                s.units_produced / max(s.step * (8 / self.TOTAL_STEPS), 0.1), 1
            ),
            cost_today_INR=s.cost_today,
            budget_today_INR=s.budget_today,
            cumulative_cost_INR=s.cumulative_cost,
            cumulative_budget_INR=s.cumulative_budget,
            active_alerts=alerts,
            crisis_type=s.active_crisis,
            crisis_severity=s.crisis_severity,
            reward=round(reward, 4),
            reward_breakdown=reward_breakdown,
            done=done,
            step=s.step,
            shift=((s.step - 1) // self.STEPS_PER_SHIFT) + 1,
            day=s.day,
            feedback=feedback
        )

    def _active_machine_counts(self) -> dict:
        """Returns {machine_id: count_online}."""
        counts = {}
        for m in self.state.factory_config.machines:
            if self.state.machine_online.get(m, True):
                counts[m] = counts.get(m, 0) + 1
        return counts

    def _base_step_cost(self) -> int:
        """Fixed overhead per step (power, admin, etc.)."""
        return 2500

    def _compute_machine_cost(self) -> int:
        """Running cost for all online machines this step."""
        total = 0
        step_fraction = 8 / self.TOTAL_STEPS   # hours per step
        for m in self.state.factory_config.machines:
            if self.state.machine_online.get(m, True):
                spec = self.machines_db.get(m, {})
                cost = spec.get("cost_per_hour_INR", 400) * step_fraction
                total += int(cost)
        return total
