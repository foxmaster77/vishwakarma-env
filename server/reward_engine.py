"""
vishwakarma_env/server/reward_engine.py

5-component reward function for the Vishwakarma environment.

Components:
  1. Production rate      (40%) — did we hit targets?
  2. Cost discipline      (25%) — did we stay in budget?
  3. Crisis response      (20%) — how well did we handle crises?
  4. Safety               (10%) — zero incidents = bonus
  5. Long-term thinking   (5%)  — buffer stock, supplier diversity
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardBreakdown:
    production: float
    cost: float
    crisis_response: float
    safety: float
    long_term: float
    total: float
    explanation: str


class RewardEngine:

    # Component weights (sum = 1.0)
    W_PRODUCTION  = 0.40
    W_COST        = 0.25
    W_CRISIS      = 0.20
    W_SAFETY      = 0.10
    W_LONGTERM    = 0.05

    # Scaling factors (max raw score per component before weighting)
    MAX_PRODUCTION  = 2.0
    MAX_COST        = 1.5
    MAX_CRISIS      = 1.5
    MAX_SAFETY      = 1.0
    MAX_LONGTERM    = 1.0

    def compute(self,
                units_produced: int,
                units_target: int,
                cost_this_step: int,
                budget_this_step: int,
                crisis_occurred: bool,
                crisis_resolved: bool,
                response_delay_steps: int,
                production_loss_ratio: float,
                safety_incidents: int,
                safety_passed: bool,
                stock_buffer_days: float,
                supplier_diversity: int,
                rerouted_successfully: bool = False) -> RewardBreakdown:

        parts = []

        # ─────────────────────────────────────
        # 1. Production (40%)
        # ─────────────────────────────────────
        prod_ratio = units_produced / max(units_target, 1)
        if prod_ratio >= 1.0:
            prod_score = self.MAX_PRODUCTION                  # full score + overage bonus
            prod_score += min((prod_ratio - 1.0) * 0.5, 0.3) # up to 0.3 bonus over target
        else:
            prod_score = prod_ratio * self.MAX_PRODUCTION     # proportional
        prod_reward = prod_score * self.W_PRODUCTION
        parts.append(f"production({prod_reward:+.3f})")

        # ─────────────────────────────────────
        # 2. Cost discipline (25%)
        # ─────────────────────────────────────
        if budget_this_step <= 0:
            cost_score = self.MAX_COST
        else:
            cost_ratio = cost_this_step / budget_this_step
            if cost_ratio <= 0.80:
                cost_score = self.MAX_COST                    # well under budget
            elif cost_ratio <= 1.0:
                cost_score = self.MAX_COST * (1 - (cost_ratio - 0.80) / 0.20 * 0.3)
            else:
                overage = cost_ratio - 1.0
                cost_score = max(-1.0, -overage * 2.0)        # penalty for overspend
        cost_reward = cost_score * self.W_COST
        parts.append(f"cost({cost_reward:+.3f})")

        # ─────────────────────────────────────
        # 3. Crisis response (20%)
        # ─────────────────────────────────────
        if not crisis_occurred:
            crisis_score = self.MAX_CRISIS * 0.5              # baseline for normal ops
        else:
            if crisis_resolved:
                speed_bonus = max(0, 1.0 - response_delay_steps * 0.2)
                containment = 1.0 - production_loss_ratio
                crisis_score = (speed_bonus * 0.7 + containment * 0.3) * self.MAX_CRISIS
                if rerouted_successfully:
                    crisis_score += 0.2                        # bonus for smart rerouting
            else:
                crisis_score = 0.0                             # unresolved = no credit
        crisis_reward = crisis_score * self.W_CRISIS
        parts.append(f"crisis({crisis_reward:+.3f})")

        # ─────────────────────────────────────
        # 4. Safety (10%)
        # ─────────────────────────────────────
        if safety_incidents > 0:
            safety_score = -safety_incidents * 2.0            # hard penalty per incident
        elif safety_passed:
            safety_score = self.MAX_SAFETY
        else:
            safety_score = 0.3                                 # warnings but no incidents
        safety_reward = safety_score * self.W_SAFETY
        parts.append(f"safety({safety_reward:+.3f})")

        # ─────────────────────────────────────
        # 5. Long-term thinking (5%)
        # ─────────────────────────────────────
        lt_score = 0.0
        if stock_buffer_days >= 3.0:
            lt_score += 0.5                                    # good buffer stock
        elif stock_buffer_days >= 1.5:
            lt_score += 0.25
        if supplier_diversity >= 2:
            lt_score += 0.3                                    # multiple suppliers
        if supplier_diversity >= 3:
            lt_score += 0.2                                    # excellent diversity
        lt_score = min(lt_score, self.MAX_LONGTERM)
        lt_reward = lt_score * self.W_LONGTERM
        parts.append(f"longterm({lt_reward:+.3f})")

        # ─────────────────────────────────────
        # Total
        # ─────────────────────────────────────
        total = round(prod_reward + cost_reward + crisis_reward +
                      safety_reward + lt_reward, 4)

        explanation = " | ".join(parts) + f" = {total:.4f}"

        return RewardBreakdown(
            production=round(prod_reward, 4),
            cost=round(cost_reward, 4),
            crisis_response=round(crisis_reward, 4),
            safety=round(safety_reward, 4),
            long_term=round(lt_reward, 4),
            total=total,
            explanation=explanation
        )

    def episode_final_reward(self,
                              total_units: int,
                              target_units: int,
                              total_cost: int,
                              total_budget: int,
                              total_crises: int,
                              crises_resolved: int,
                              safety_incidents: int,
                              avg_buffer_days: float) -> float:
        """
        End-of-episode bonus/penalty on top of cumulative step rewards.
        """
        bonus = 0.0

        # Hit daily target bonus
        if total_units >= target_units:
            bonus += 2.0

        # Under budget bonus
        if total_cost <= total_budget * 0.90:
            bonus += 1.0

        # Perfect crisis management
        if total_crises > 0 and crises_resolved == total_crises:
            bonus += 1.5

        # Zero safety incidents bonus
        if safety_incidents == 0:
            bonus += 1.0

        # Good buffer stock management
        if avg_buffer_days >= 2.0:
            bonus += 0.5

        # Penalty: missed target badly
        if total_units < target_units * 0.70:
            bonus -= 2.0

        # Penalty: safety incidents
        bonus -= safety_incidents * 0.5

        return round(bonus, 4)
