"""
vishwakarma_env/server/multi_agent_environment.py

Two factories share one supplier pool.
Agent must allocate scarce stock between them during supply shocks.
Factory A (auto, Rs280/unit) vs Factory B (textile, Rs200/unit)
Smart allocation = prioritize higher margin factory.
"""

import uuid, random
from dataclasses import dataclass, field
from typing import Optional, List
from ..models import VishwakarmaAction
from .vishwakarma_environment import VishwakarmaEnvironment


@dataclass
class MultiAgentAction:
    factory_a: VishwakarmaAction = field(
        default_factory=lambda: VishwakarmaAction(directive="run_normal"))
    factory_b: VishwakarmaAction = field(
        default_factory=lambda: VishwakarmaAction(directive="run_normal"))
    stock_split_to_a: float = 0.5   # 0.0–1.0 fraction of shared stock to factory A
    reasoning: str = ""


@dataclass
class MultiAgentObservation:
    factory_a: object = None
    factory_b: object = None
    supplier_stock_tons: float = 20.0
    supplier_under_shock: bool = False
    shock_message: str = ""
    combined_reward: float = 0.0
    combined_units_produced: int = 0
    combined_units_target: int = 0
    step: int = 0
    done: bool = False
    feedback: str = ""


class MultiAgentVishwakarmaEnvironment:
    """
    Two-factory shared-supplier environment.
    The ONLY multi-agent OpenEnv environment in this hackathon.

    Core challenge: during a supply shock, the agent must decide
    how to split limited steel stock between two factories with
    different margins. Optimal = prioritize higher margin factory.
    """

    TOTAL_STEPS         = 16
    MARGIN_A            = 280    # Rs/unit (auto components)
    MARGIN_B            = 200    # Rs/unit (textile machinery)
    DAILY_SUPPLY        = 12.0   # tons/day total capacity
    SHOCK_PROB_PER_STEP = 0.30   # after step 4

    def __init__(self, seed: int = None):
        self.seed  = seed
        self._rng  = random.Random(seed)
        self.env_a = VishwakarmaEnvironment("auto_components_pune",  seed=seed)
        self.env_b = VishwakarmaEnvironment("textile_mill_surat",
                                             seed=(seed or 0) + 100)
        self._step            = 0
        self._shock_active    = False
        self._shock_steps_left = 0
        self._shared_stock    = self.DAILY_SUPPLY
        self._cumulative_reward = 0.0
        self._alloc_scores: List[float] = []
        self._episode_id      = ""

    # ── reset() ──────────────────────────────────────────

    def reset(self) -> MultiAgentObservation:
        obs_a = self.env_a.reset()
        obs_b = self.env_b.reset()
        self._step             = 0
        self._shock_active     = False
        self._shock_steps_left = 0
        self._shared_stock     = self.DAILY_SUPPLY
        self._cumulative_reward = 0.0
        self._alloc_scores     = []
        self._episode_id       = str(uuid.uuid4())[:8]
        return self._obs(obs_a, obs_b, 0.0, False,
            f"Multi-factory episode {self._episode_id}.\n"
            f"A (Pune Auto):    target {obs_a.units_target_today} units @ Rs{self.MARGIN_A}/unit\n"
            f"B (Surat Textile): target {obs_b.units_target_today} units @ Rs{self.MARGIN_B}/unit\n"
            f"Shared supplier: {self.DAILY_SUPPLY}t/day. Allocate wisely during shocks."
        )

    # ── step() ───────────────────────────────────────────

    def step(self, action: MultiAgentAction) -> MultiAgentObservation:
        self._step += 1
        lines = []
        alloc_bonus = 0.0

        # ── Supplier shock lifecycle ──────────────────────
        if self._shock_active:
            self._shock_steps_left -= 1
            if self._shock_steps_left <= 0:
                self._shock_active = False
                self._shared_stock = self.DAILY_SUPPLY
                lines.append("✓ Supplier shock resolved — normal supply restored.")
        elif self._step > 4 and self._rng.random() < self.SHOCK_PROB_PER_STEP:
            self._shock_active     = True
            self._shock_steps_left = self._rng.randint(2, 4)
            self._shared_stock     = self._rng.uniform(2.5, 5.0)
            reason = self._rng.choice([
                "Tata Steel Pune furnace breakdown — 40% capacity only",
                "Transport blockade NH-48 — emergency allocation required",
                "Mill workers strike — limited delivery possible",
                "Floods near Khopoli — restricted truck movement",
            ])
            lines.append(
                f"🚨 SUPPLIER SHOCK: {reason}\n"
                f"   Available: {self._shared_stock:.1f}t shared between A+B.\n"
                f"   Factory A margin Rs{self.MARGIN_A}/unit  "
                f"Factory B margin Rs{self.MARGIN_B}/unit.\n"
                f"   Optimal split: {self.MARGIN_A/(self.MARGIN_A+self.MARGIN_B):.0%} to A."
            )

        # ── Stock allocation during shock ─────────────────
        if self._shock_active:
            split   = max(0.0, min(1.0, action.stock_split_to_a))
            s_to_a  = self._shared_stock * split
            s_to_b  = self._shared_stock * (1.0 - split)

            # Optimal fraction = margin_A / (margin_A + margin_B)
            optimal = self.MARGIN_A / (self.MARGIN_A + self.MARGIN_B)
            error   = abs(split - optimal)
            score   = max(0.0, 1.0 - error * 3.0)
            self._alloc_scores.append(score)
            alloc_bonus = score * 2.0

            # Inject into factories
            if self.env_a.state:
                self.env_a.state.stock_tons = max(self.env_a.state.stock_tons, s_to_a)
            if self.env_b.state:
                self.env_b.state.stock_tons = max(self.env_b.state.stock_tons, s_to_b)

            lines.append(
                f"  Allocation: {s_to_a:.1f}t→A ({split:.0%}), "
                f"{s_to_b:.1f}t→B ({1-split:.0%})  "
                f"score={score:.2f}/1.00"
            )

        # ── Step both factories ───────────────────────────
        action.factory_a.reasoning = action.reasoning
        action.factory_b.reasoning = action.reasoning
        obs_a = self.env_a.step(action.factory_a)
        obs_b = self.env_b.step(action.factory_b)

        # ── Combined reward ───────────────────────────────
        rev_a  = obs_a.units_produced_today * self.MARGIN_A
        rev_b  = obs_b.units_produced_today * self.MARGIN_B
        tgt_rev = (obs_a.units_target_today * self.MARGIN_A +
                   obs_b.units_target_today * self.MARGIN_B)
        rev_eff = (rev_a + rev_b) / max(tgt_rev, 1)
        rev_bonus = (rev_eff - 0.5) * 1.0

        reward = round(obs_a.reward + obs_b.reward + alloc_bonus + rev_bonus, 4)
        self._cumulative_reward += reward

        # ── Done ─────────────────────────────────────────
        done = self._step >= self.TOTAL_STEPS
        if done:
            ep = self._ep_bonus(obs_a, obs_b)
            reward += ep
            self._cumulative_reward += ep
            avg_alloc = (sum(self._alloc_scores) /
                         max(len(self._alloc_scores), 1))
            lines.append(
                f"\n{'='*54}\nMULTI-FACTORY COMPLETE\n"
                f"  A: {obs_a.units_produced_today}/{obs_a.units_target_today}  "
                f"B: {obs_b.units_produced_today}/{obs_b.units_target_today}\n"
                f"  Allocation score: {avg_alloc:.2f}/1.00\n"
                f"  Ep bonus: {ep:+.2f}\n"
                f"  TOTAL: {self._cumulative_reward:.4f}\n{'='*54}"
            )

        lines.append(
            f"Step {self._step:02d}/{self.TOTAL_STEPS} | "
            f"A:{obs_a.units_produced_today}/{obs_a.units_target_today} "
            f"B:{obs_b.units_produced_today}/{obs_b.units_target_today} | "
            f"reward={reward:+.4f}"
        )

        return self._obs(obs_a, obs_b, reward, done, "\n".join(lines))

    def state_info(self) -> dict:
        return {
            "episode_id":        self._episode_id,
            "step":              self._step,
            "total_steps":       self.TOTAL_STEPS,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "factory_a_units":   self.env_a.state.units_produced if self.env_a.state else 0,
            "factory_b_units":   self.env_b.state.units_produced if self.env_b.state else 0,
            "allocation_decisions": len(self._alloc_scores),
            "avg_allocation_score": round(
                sum(self._alloc_scores) / max(len(self._alloc_scores), 1), 3),
        }

    def _ep_bonus(self, obs_a, obs_b) -> float:
        b = 0.0
        if obs_a.units_produced_today >= obs_a.units_target_today: b += 2.0
        if obs_b.units_produced_today >= obs_b.units_target_today: b += 2.0
        if self._alloc_scores:
            b += (sum(self._alloc_scores) / len(self._alloc_scores)) * 3.0
        return round(b, 4)

    def _obs(self, obs_a, obs_b, reward, done, feedback) -> MultiAgentObservation:
        return MultiAgentObservation(
            factory_a=obs_a, factory_b=obs_b,
            supplier_stock_tons=round(self._shared_stock, 2),
            supplier_under_shock=self._shock_active,
            combined_reward=round(reward, 4),
            combined_units_produced=(
                (obs_a.units_produced_today if obs_a else 0) +
                (obs_b.units_produced_today if obs_b else 0)
            ),
            combined_units_target=(
                (obs_a.units_target_today if obs_a else 0) +
                (obs_b.units_target_today if obs_b else 0)
            ),
            step=self._step, done=done, feedback=feedback,
        )
