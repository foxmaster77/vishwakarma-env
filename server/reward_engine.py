"""
vishwakarma_env/server/reward_engine.py  v3
"""

import re
from dataclasses import dataclass


@dataclass
class RewardBreakdown:
    production: float
    cost: float
    crisis_response: float
    safety: float
    long_term: float
    reasoning: float
    total: float
    explanation: str


class RewardEngine:

    def compute(self,
                units_produced: int, units_target: int,
                production_multiplier: float,
                cost_this_step: int, budget_this_step: int,
                crisis_active: bool,
                crisis_status: str,       # "new"|"resolved"|"expired"|"active"|"none"
                crisis_severity: str,     # "low"|"medium"|"high"
                response_was_correct: bool,
                safety_passed: bool, safety_incidents: int,
                stock_buffer_days: float, supplier_diversity: int,
                reasoning: str = "",
                observation_context: dict = None,
                reasoning_score: float = None) -> RewardBreakdown:

        sev_mult = {"low": 0.8, "medium": 1.3, "high": 2.0}.get(crisis_severity, 1.0)

        # 1. Production (-1.0 to +2.3)
        prod_ratio = (units_produced / max(units_target, 1)) * production_multiplier
        if prod_ratio >= 1.0:
            prod = 1.0 + min((prod_ratio - 1.0) * 0.5, 0.3)
        elif prod_ratio >= 0.85: prod = 0.7
        elif prod_ratio >= 0.60: prod = 0.4
        elif prod_ratio >= 0.40: prod = 0.1
        else:                    prod = -0.5

        # 2. Cost (-1.5 to +1.0) — lenient during crisis
        if budget_this_step <= 0:
            cost = 0.5
        else:
            r = cost_this_step / budget_this_step
            if crisis_active:
                cost = 0.5 if r <= 1.0 else (0.2 if r <= 2.0 else -0.5)
            else:
                if r <= 0.80:   cost = 1.0
                elif r <= 1.0:  cost = 0.5
                elif r <= 1.2:  cost = -0.3
                else:           cost = -1.0

        # 3. Crisis response (-4.0 to +5.0) — main differentiator
        if crisis_status == "resolved":
            crisis = 2.5 * sev_mult
        elif crisis_status == "expired":
            crisis = -2.0 * sev_mult
        elif crisis_status == "active":
            crisis = 0.3 * sev_mult if response_was_correct else -0.5 * sev_mult
        elif crisis_status == "new":
            crisis = 0.0
        else:  # "none"
            crisis = 0.2

        # 4. Safety (-2.0 to +0.5)
        if safety_incidents > 0:
            safety = -2.0 * safety_incidents
        elif safety_passed:
            safety = 0.5
        else:
            safety = 0.1

        # 5. Long-term (-0.5 to +1.0)
        lt = 0.0
        if stock_buffer_days >= 3.0:   lt += 0.5
        elif stock_buffer_days >= 1.5: lt += 0.25
        elif stock_buffer_days < 0.5:  lt -= 0.3
        if supplier_diversity >= 2:    lt += 0.3
        lt = max(-0.5, min(lt, 1.0))

        # 6. Reasoning (0 to +0.5) — pre-scored by ReasoningScorer in environment
        rsn = reasoning_score if reasoning_score is not None else self._score_reasoning(
            reasoning, crisis_status, crisis_severity,
            stock_buffer_days, observation_context or {}
        )

        total = round(prod + cost + crisis + safety + lt + rsn, 4)
        explanation = (
            f"prod={prod:+.2f} cost={cost:+.2f} "
            f"crisis={crisis:+.2f}[{crisis_status}] "
            f"safety={safety:+.2f} lt={lt:+.2f} rsn={rsn:+.2f} = {total:.4f}"
        )

        return RewardBreakdown(
            production=round(prod, 4), cost=round(cost, 4),
            crisis_response=round(crisis, 4), safety=round(safety, 4),
            long_term=round(lt, 4), reasoning=round(rsn, 4),
            total=total, explanation=explanation
        )

    def _score_reasoning(self, reasoning, crisis_status, crisis_severity,
                         stock_days, context) -> float:
        if not reasoning or len(reasoning.strip()) < 10:
            return 0.0
        score = 0.0
        r = reasoning.lower()
        if len(r.split()) >= 15:
            score += 0.1
        if crisis_status in ("active", "new", "resolved"):
            crisis_words = ["crisis","breakdown","shortage","absent","supply",
                            "maintenance","reroute","contractor","stock","overtime",
                            "quality","reject","demand","spike","order"]
            if any(w in r for w in crisis_words):
                score += 0.15
        if re.search(r'\d+', r):
            score += 0.05
        if stock_days < 1.5 and any(w in r for w in ["stock","material","supplier","order"]):
            score += 0.10
        if crisis_severity == "high" and any(w in r for w in
                ["urgent","critical","severe","high","emergency","immediately"]):
            score += 0.10
        return min(score, 0.5)

    def episode_final_reward(self, units_produced, units_target,
                              total_cost, total_budget,
                              crises_total, crises_resolved, crises_expired,
                              safety_incidents, avg_buffer_days) -> float:
        bonus = 0.0
        ratio = units_produced / max(units_target, 1)
        if ratio >= 1.0:   bonus += 3.0
        elif ratio >= 0.90: bonus += 1.5
        elif ratio >= 0.75: bonus += 0.5
        else:               bonus -= 1.0

        if total_budget > 0:
            cr = total_cost / total_budget
            if cr <= 0.90:   bonus += 1.0
            elif cr <= 1.05: bonus += 0.3
            else:            bonus -= 0.5

        if crises_total > 0:
            bonus += (crises_resolved / crises_total) * 3.0
            bonus -= crises_expired * 1.5

        if safety_incidents == 0:
            bonus += 1.0
        else:
            bonus -= safety_incidents * 0.5

        if avg_buffer_days >= 2.0:
            bonus += 0.5

        return round(bonus, 4)
