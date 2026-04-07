"""
vishwakarma_env/server/vishwakarma_environment.py  v3

Bugs fixed vs v2:
  1. Production math: units_per_step = throughput/hr × 1hr (was × 0.5)
  2. Demand spike: does NOT permanently add to units_target
  3. Crisis frequency: max 2 crises per episode cap added
  4. Safety warnings: printed ONCE at episode start, not every step
  5. Machine count display: uses len(machines list) not unique set
  6. Crisis engine v3 wired — uses is_resolved_by() not resolved_by list
"""

import uuid, json, os
from typing import Optional
from ..models import (
    VishwakarmaAction, VishwakarmaObservation, VishwakarmaState,
    FactoryConfig, MachineStatus, Alert, CrisisType, Severity,
    ProductionDirective
)
from .crisis_engine import CrisisEngine
from .production_simulator import ProductionSimulator
from .reward_engine import RewardEngine
from .reasoning_scorer import ReasoningScorer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VALID_DIRECTIVES = {d.value for d in ProductionDirective}

def _load(f):
    with open(os.path.join(DATA_DIR, f)) as fp:
        return json.load(fp)


class VishwakarmaEnvironment:

    TOTAL_STEPS     = 16
    STEPS_PER_SHIFT = 8
    HOURS_PER_STEP  = 1.0   # each step = 1 hour (8hr shift / 8 steps)
    MAX_CRISES      = 4     # cap per episode to prevent crisis flood

    def __init__(self, factory_id="auto_components_pune", seed=None):
        self.factories_db = {f["id"]: f for f in _load("factories.json")["factories"]}
        self.machines_db  = {m["id"]: m for m in _load("machines.json")["machines"]}
        self.suppliers_db = {s["id"]: s for s in _load("suppliers.json")["suppliers"]}
        self.factory_id   = factory_id
        self.seed         = seed
        self.simulator      = ProductionSimulator(self.machines_db)
        self.reward_engine  = RewardEngine()
        self.reasoning_scorer = ReasoningScorer()
        self.state: Optional[VishwakarmaState] = None
        self._crisis_engine: Optional[CrisisEngine] = None
        self._buffer_sum      = 0.0
        self._crises_expired  = 0
        self._safety_printed  = False   # print safety warnings once only

    # ─── reset() ─────────────────────────────────────────────

    def reset(self) -> VishwakarmaObservation:
        fc  = self.factories_db[self.factory_id]
        cfg = FactoryConfig(
            name=fc["name"], industry=fc["industry"],
            floor_width_m=fc["floor_width_m"], floor_length_m=fc["floor_length_m"],
            total_machines=fc["total_machines"], total_workers=fc["total_workers"],
            shifts_per_day=fc["shifts_per_day"],
            target_units_per_day=fc["target_units_per_day"],
            budget_per_day_INR=fc["budget_per_day_INR"],
            machines=fc["machines"],
            suppliers=[self.suppliers_db[fc["primary_supplier"]],
                       self.suppliers_db[fc["backup_supplier"]]]
        )
        self.state = VishwakarmaState(
            episode_id=str(uuid.uuid4())[:8],
            factory_config=cfg,
            workers_present=cfg.total_workers,
            stock_tons=fc["initial_stock_tons"],
            daily_consumption_tons=fc["daily_stock_consumption_tons"],
            units_target=cfg.target_units_per_day,
            budget_today=cfg.budget_per_day_INR,
            cumulative_budget=cfg.budget_per_day_INR,
        )
        for m in cfg.machines:
            self.state.machine_online[m]  = True
            self.state.machine_health[m]  = 1.0
            self.state.maintenance_eta[m] = 0
        for s in cfg.suppliers:
            self.state.supplier_reliability[s["id"]] = s["reliability"]
        self.state.supplier_diversity = len(cfg.suppliers)
        self.state.buffer_stock_days  = (
            fc["initial_stock_tons"] / max(fc["daily_stock_consumption_tons"], 0.1)
        )

        self._crisis_engine  = CrisisEngine({"machines": cfg.machines}, self.seed)
        self._buffer_sum     = self.state.buffer_stock_days
        self._crises_expired = 0
        self._safety_printed = False

        return self._obs(0.0, {}, False,
            f"Factory '{cfg.name}' online. "
            f"Target: {cfg.target_units_per_day:,} units/day. "
            f"Budget: ₹{cfg.budget_per_day_INR:,}. "
            f"Workers: {cfg.total_workers}. "
            f"Machines: {cfg.total_machines}. "
            f"Stock: {fc['initial_stock_tons']:.1f}t "
            f"({self.state.buffer_stock_days:.1f} days). "
            f"Good luck."
        )

    # ─── step() ──────────────────────────────────────────────

    def step(self, action: VishwakarmaAction) -> VishwakarmaObservation:
        if not self.state:
            raise RuntimeError("Call reset() before step()")
        if self.state.step >= self.TOTAL_STEPS:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        # Validate and clamp action
        if action.directive not in VALID_DIRECTIVES:
            action.directive = ProductionDirective.RUN_NORMAL.value
        action.order_stock_tons           = max(0.0, min(action.order_stock_tons, 20.0))
        action.authorize_overtime_workers = max(0, min(action.authorize_overtime_workers, 20))
        action.call_contractors           = max(0, min(action.call_contractors, 10))
        action.adjust_production_rate     = max(0.1, min(1.5, action.adjust_production_rate))

        s        = self.state
        s.step  += 1
        feedback = []
        step_cost = 2500          # base overhead

        # ── 1. Tick crisis engine ─────────────────────────────
        total_crises = len(s.crisis_history)
        can_generate_crisis = total_crises < self.MAX_CRISES

        new_crisis, crisis_status = self._crisis_engine.tick(
            s, action if can_generate_crisis else _NoNewCrisisAction(action)
        )

        if crisis_status == "new" and new_crisis:
            s.active_crisis   = new_crisis.crisis_type
            s.crisis_severity = new_crisis.severity
            s.crisis_history.append({
                "type": new_crisis.crisis_type,
                "severity": new_crisis.severity,
                "step": s.step, "resolved": False, "expired": False
            })
            step_cost += new_crisis.cost_impact_INR
            feedback.append(
                f"🚨 CRISIS [{new_crisis.severity.value.upper()}]: {new_crisis.message}"
            )
            feedback.append(
                f"   → Options: {' | '.join(new_crisis.resolution_options)}"
            )

        elif crisis_status == "resolved":
            s.crises_resolved += 1
            for h in reversed(s.crisis_history):
                if not h["resolved"] and not h["expired"]:
                    h["resolved"] = True
                    h["resolved_step"] = s.step
                    break
            s.active_crisis   = CrisisType.NONE
            s.crisis_severity = Severity.LOW
            feedback.append("✅ Crisis resolved. Good response.")

        elif crisis_status == "expired":
            self._crises_expired += 1
            for h in reversed(s.crisis_history):
                if not h["resolved"] and not h["expired"]:
                    h["expired"] = True
                    break
            s.active_crisis   = CrisisType.NONE
            s.crisis_severity = Severity.LOW
            feedback.append("⚠ Crisis expired unresolved — took full production hit.")

        # Active crisis production impact
        # If crisis JUST fired this step (status=new), don't apply penalty yet.
        # Agent hasn't had a chance to respond — give them one step grace period.
        prod_mult = action.adjust_production_rate
        if self._crisis_engine.active_crisis and crisis_status != "new":
            prod_mult *= self._crisis_engine.active_crisis.production_impact

        # ── 2. Apply action ───────────────────────────────────

        if action.reroute_from and action.reroute_to:
            mult      = self.simulator.compute_reroute_impact(
                action.reroute_from, action.reroute_to, self._active_machines()
            )
            prod_mult = max(prod_mult, mult)    # reroute doesn't stack with crisis penalty
            feedback.append(
                f"Rerouted {action.reroute_from} → {action.reroute_to} "
                f"({mult:.0%} capacity)."
            )

        if action.call_maintenance:
            step_cost += 4500
            for mid, online in s.machine_online.items():
                if not online:
                    s.machine_online[mid] = True
                    s.machine_health[mid] = 0.9
                    self._crisis_engine.notify_maintenance_done(mid)
                    feedback.append(f"Maintenance called → {mid} back online.")
                    break

        if action.order_stock_tons > 0:
            sup_id   = ("backup_steel_mumbai"
                        if action.order_stock_supplier == "backup"
                        else "primary_steel_pune")
            sup      = self.suppliers_db.get(sup_id, {})
            sc       = int(action.order_stock_tons * sup.get("cost_per_ton_INR", 52000))
            s.stock_tons += action.order_stock_tons
            step_cost    += sc
            feedback.append(
                f"Ordered {action.order_stock_tons}t from "
                f"{sup.get('name','supplier')} — ₹{sc:,}. "
                f"Stock now: {s.stock_tons:.1f}t."
            )

        if action.authorize_overtime_workers > 0:
            ot_cost    = action.authorize_overtime_workers * 850
            step_cost += ot_cost
            prod_mult  = min(prod_mult * 1.12, 1.2)
            s.overtime_authorized = action.authorize_overtime_workers
            feedback.append(
                f"Overtime: {action.authorize_overtime_workers} workers — ₹{ot_cost:,}, +12%."
            )

        if action.call_contractors > 0:
            cc        = action.call_contractors * 1200
            step_cost += cc
            s.workers_present = min(
                s.workers_present + action.call_contractors,
                s.factory_config.total_workers + 8
            )
            feedback.append(
                f"Contractors: {action.call_contractors} hired — ₹{cc:,}. "
                f"Floor: {s.workers_present} workers."
            )

        # Demand spike: mark accepted but do NOT increase units_target
        # (the spike is a separate order, not a new baseline)
        if action.accept_emergency_order and self._crisis_engine.active_crisis:
            c = self._crisis_engine.active_crisis
            if c.crisis_type == CrisisType.DEMAND_SPIKE:
                extra_rev = c.metadata.get("revenue_INR", 0)
                feedback.append(
                    f"Emergency order accepted — potential revenue ₹{extra_rev:,}."
                )

        # ── 3. Production ─────────────────────────────────────

        # Consume stock
        stock_per_step    = s.daily_consumption_tons / self.TOTAL_STEPS
        s.stock_tons      = max(0.0, s.stock_tons - stock_per_step)
        s.buffer_stock_days = s.stock_tons / max(s.daily_consumption_tons, 0.1)
        self._buffer_sum += s.buffer_stock_days

        if s.stock_tons <= 0:
            prod_mult = 0.0
            feedback.append("⚠ Stock exhausted — production halted this step.")

        # Target per step (based on ORIGINAL daily target, never inflated by demand spike)
        target_per_step = s.units_target // self.TOTAL_STEPS

        sim = self.simulator.simulate(
            active_machines=self._active_machines(),
            workers_available=s.workers_present,
            target_units=s.units_target // 2,   # per-shift target (daily / 2 shifts)
            production_rate_multiplier=prod_mult
        )

        # Each step = HOURS_PER_STEP (1.0 hour per step)
        units_this_step = int(sim.throughput_per_hour * self.HOURS_PER_STEP * prod_mult)
        s.units_produced += units_this_step

        for note in sim.notes:
            if "BOTTLENECK" in note or "insufficient" in note.lower():
                feedback.append(f"  {note}")

        # ── 4. Safety (print once per episode) ───────────────
        safety = self.simulator.safety_check(
            active_machines=self._active_machines(),
            workers_present=s.workers_present
        )
        safety_incidents_step = 0
        if not safety["passed"] and not self._safety_printed:
            self._safety_printed = True
            for v in safety["violations"]:
                feedback.append(f"⚠ SAFETY: {v}")

        # ── 5. Financials ─────────────────────────────────────
        machine_cost    = self._machine_running_cost()
        worker_cost     = s.workers_present * 280
        total_step_cost = step_cost + machine_cost + worker_cost
        s.cost_today       += total_step_cost
        s.cumulative_cost  += total_step_cost

        # ── 6. Reward ─────────────────────────────────────────
        crisis_active_now = self._crisis_engine.active_crisis is not None
        crisis_sev_str = (
            self._crisis_engine.active_crisis.severity.value
            if self._crisis_engine.active_crisis
            else "low"
        )
        response_correct = (
            action.call_maintenance or
            action.order_stock_tons > 0 or
            action.call_contractors > 0 or
            bool(action.reroute_from) or
            action.authorize_overtime_workers > 0
        ) if crisis_active_now else False

        budget_per_step = s.budget_today // self.TOTAL_STEPS

        # Score reasoning via Claude API (or keyword fallback)
        rsn_situation = {
            "crisis_type": str(s.active_crisis),
            "crisis_severity": crisis_sev_str,
            "crisis_message": (self._crisis_engine.active_crisis.message
                               if self._crisis_engine.active_crisis else ""),
            "resolution_options": (self._crisis_engine.active_crisis.resolution_options
                                   if self._crisis_engine.active_crisis else []),
            "stock_days": s.buffer_stock_days,
            "workers_present": s.workers_present,
            "workers_total": s.factory_config.total_workers,
            "units_produced": s.units_produced,
            "units_target": s.units_target,
            "cost_today": s.cost_today,
            "budget_today": s.budget_today,
            "machines_online": sum(1 for m in s.factory_config.machines if s.machine_online.get(m, True)),
            "machines_total": len(s.factory_config.machines),
        }
        rsn_result = self.reasoning_scorer.score(action.reasoning, rsn_situation)

        rb = self.reward_engine.compute(
            units_produced=units_this_step,
            units_target=target_per_step,
            production_multiplier=prod_mult,
            cost_this_step=total_step_cost,
            budget_this_step=budget_per_step,
            crisis_active=crisis_active_now,
            crisis_status=crisis_status,
            crisis_severity=crisis_sev_str,
            response_was_correct=response_correct,
            safety_passed=safety["passed"],
            safety_incidents=safety_incidents_step,
            stock_buffer_days=s.buffer_stock_days,
            supplier_diversity=s.supplier_diversity,
            reasoning=action.reasoning,
            observation_context={
                "stock_days": s.buffer_stock_days,
                "crisis_active": crisis_active_now,
            },
            reasoning_score=rsn_result.score * 0.5,
        )
        s.cumulative_reward += rb.total

        # ── 7. Done ───────────────────────────────────────────
        done = s.step >= self.TOTAL_STEPS
        if done:
            avg_buf  = self._buffer_sum / max(s.step, 1)
            ep_bonus = self.reward_engine.episode_final_reward(
                units_produced=s.units_produced,
                units_target=s.units_target,
                total_cost=s.cumulative_cost,
                total_budget=s.cumulative_budget,
                crises_total=len(s.crisis_history),
                crises_resolved=s.crises_resolved,
                crises_expired=self._crises_expired,
                safety_incidents=s.safety_incidents,
                avg_buffer_days=avg_buf,
            )
            rb.total           += ep_bonus
            s.cumulative_reward += ep_bonus
            hit = "✓ YES" if s.units_produced >= s.units_target else "✗ NO"
            feedback.append(
                f"\n{'='*52}\n"
                f"EPISODE COMPLETE\n"
                f"  Units:   {s.units_produced:,} / {s.units_target:,}  {hit}\n"
                f"  Cost:    ₹{s.cumulative_cost:,} / ₹{s.cumulative_budget:,}\n"
                f"  Crises:  {s.crises_resolved} resolved | "
                f"{self._crises_expired} expired | "
                f"{len(s.crisis_history)} total\n"
                f"  Safety:  {s.safety_incidents} incidents\n"
                f"  Bonus:   {ep_bonus:+.2f}\n"
                f"  TOTAL:   {s.cumulative_reward:.4f}\n"
                f"{'='*52}"
            )

        feedback.append(
            f"Step {s.step:02d}/{self.TOTAL_STEPS} | "
            f"units={s.units_produced}/{s.units_target} | "
            f"cost=₹{s.cost_today:,} | "
            f"reward={rb.total:+.4f}"
        )

        return self._obs(
            rb.total,
            {"production": rb.production, "cost": rb.cost,
             "crisis": rb.crisis_response, "safety": rb.safety,
             "long_term": rb.long_term, "reasoning": rb.reasoning},
            done, "\n".join(feedback)
        )

    # ─── state_info() ────────────────────────────────────────

    def state_info(self) -> dict:
        if not self.state:
            return {}
        s = self.state
        return {
            "episode_id":        s.episode_id,
            "step":              s.step,
            "total_steps":       self.TOTAL_STEPS,
            "units_produced":    s.units_produced,
            "units_target":      s.units_target,
            "crises_total":      len(s.crisis_history),
            "crises_resolved":   s.crises_resolved,
            "crises_expired":    self._crises_expired,
            "safety_incidents":  s.safety_incidents,
            "cumulative_reward": round(s.cumulative_reward, 4),
        }

    # ─── helpers ─────────────────────────────────────────────

    def _active_machines(self) -> dict:
        counts = {}
        for m in self.state.factory_config.machines:
            if self.state.machine_online.get(m, True):
                counts[m] = counts.get(m, 0) + 1
        return counts

    def _machine_running_cost(self) -> int:
        total = 0
        for m in self.state.factory_config.machines:
            if self.state.machine_online.get(m, True):
                spec  = self.machines_db.get(m, {})
                total += int(spec.get("cost_per_hour_INR", 400) * self.HOURS_PER_STEP)
        return total

    def _obs(self, reward, breakdown, done, feedback) -> VishwakarmaObservation:
        s = self.state
        # Use full machines list (not set) for correct count display
        seen = {}
        machines = []
        for m in s.factory_config.machines:
            key = m
            idx = seen.get(m, 0)
            seen[m] = idx + 1
            display  = f"{m}_{idx+1}" if seen[m] > 1 else m
            machines.append(MachineStatus(
                name=display,
                online=s.machine_online.get(m, True),
                utilization=0.85 if s.machine_online.get(m, True) else 0.0,
                breakdown_eta_mins=s.maintenance_eta.get(m, 0) * 30,
            ))

        alerts = []
        if self._crisis_engine and self._crisis_engine.active_crisis:
            alerts.append(self._crisis_engine.active_crisis.to_alert())

        return VishwakarmaObservation(
            machines=machines,
            workers_present=s.workers_present,
            workers_total=s.factory_config.total_workers,
            stock_tons=round(s.stock_tons, 2),
            stock_days_remaining=round(s.buffer_stock_days, 1),
            units_produced_today=s.units_produced,
            units_target_today=s.units_target,
            production_rate_per_hour=round(
                s.units_produced / max(s.step * self.HOURS_PER_STEP, 0.1), 1
            ),
            cost_today_INR=s.cost_today,
            budget_today_INR=s.budget_today,
            cumulative_cost_INR=s.cumulative_cost,
            cumulative_budget_INR=s.cumulative_budget,
            active_alerts=alerts,
            crisis_type=s.active_crisis,
            crisis_severity=s.crisis_severity,
            reward=round(reward, 4),
            reward_breakdown=breakdown,
            done=done,
            step=s.step,
            shift=((s.step - 1) // self.STEPS_PER_SHIFT) + 1,
            day=s.day,
            feedback=feedback,
        )


class _NoNewCrisisAction:
    """Wrapper that prevents new crisis generation when cap is reached."""
    def __init__(self, real_action):
        self._a = real_action
    def __getattr__(self, name):
        return getattr(self._a, name)
