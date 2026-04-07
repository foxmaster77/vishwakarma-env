"""
vishwakarma_env/server/reasoning_scorer.py

Scores the agent's chain-of-thought reasoning using Claude API.
Called inside the reward engine when ANTHROPIC_API_KEY is set.
Falls back to keyword scoring if API unavailable.

Usage:
    scorer = ReasoningScorer()
    score = scorer.score(reasoning, situation)   # returns 0.0–1.0
"""

import os
import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoringResult:
    score: float          # 0.0 – 1.0
    method: str           # "api" or "keyword"
    explanation: str


class ReasoningScorer:
    """
    Scores agent reasoning quality (0.0–1.0) using Claude API.
    Falls back to keyword scoring if API key not set or call fails.

    Integrated into RewardEngine.compute() as the reasoning component.
    Max reward contribution: 0.5 points (reasoning_score * 0.5).
    """

    SYSTEM_PROMPT = """You are evaluating an AI agent's reasoning quality in a factory management simulation.

The agent must manage an Indian manufacturing plant, responding to crises like machine breakdowns, supply shocks, worker absenteeism, quality failures, and demand spikes.

You will be given:
1. The current factory situation (what the agent can observe)
2. The agent's reasoning text explaining its decision

Score the reasoning from 0.0 to 1.0 based on:
- Does it correctly identify the crisis type? (0.2)
- Does it reference specific observed data (stock levels, worker counts, costs)? (0.2)
- Is the proposed action appropriate for the crisis? (0.3)
- Is the reasoning logically coherent and specific? (0.3)

Return ONLY a JSON object with exactly this format:
{"score": 0.75, "explanation": "one sentence why"}

Do not include anything else. No markdown, no preamble."""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._available = bool(self.api_key)
        self._cache: dict = {}   # simple cache to avoid duplicate calls

    def score(self, reasoning: str, situation: dict) -> ScoringResult:
        """
        Score reasoning quality against the current factory situation.

        situation dict keys:
          crisis_type, crisis_severity, crisis_message,
          stock_days, workers_present, workers_total,
          units_produced, units_target, cost_today, budget_today,
          machines_online, machines_total
        """
        if not reasoning or len(reasoning.strip()) < 5:
            return ScoringResult(0.0, "keyword", "No reasoning provided")

        # Cache key — avoid re-scoring identical inputs
        cache_key = f"{reasoning[:50]}|{situation.get('crisis_type','none')}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._available:
            result = self._score_via_api(reasoning, situation)
        else:
            result = self._score_via_keywords(reasoning, situation)

        self._cache[cache_key] = result
        return result

    # ─────────────────────────────────────────
    # Claude API scoring
    # ─────────────────────────────────────────

    def _score_via_api(self, reasoning: str, situation: dict) -> ScoringResult:
        try:
            import urllib.request

            situation_text = self._format_situation(situation)
            user_message = (
                f"FACTORY SITUATION:\n{situation_text}\n\n"
                f"AGENT REASONING:\n{reasoning}"
            )

            payload = json.dumps({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 150,
                "system": self.SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_message}]
            }).encode()

            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=payload,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())

            text = data["content"][0]["text"].strip()
            parsed = json.loads(text)
            score = float(max(0.0, min(1.0, parsed["score"])))
            explanation = parsed.get("explanation", "")

            return ScoringResult(score, "api", explanation)

        except Exception as e:
            # API failed — fall back to keyword scoring silently
            return self._score_via_keywords(reasoning, situation)

    # ─────────────────────────────────────────
    # Keyword fallback scoring
    # ─────────────────────────────────────────

    def _score_via_keywords(self, reasoning: str, situation: dict) -> ScoringResult:
        score = 0.0
        r = reasoning.lower()
        crisis_type = situation.get("crisis_type", "none").lower()
        crisis_active = crisis_type not in ("none", "crisistype.none", "")

        # Length — non-trivial reasoning
        word_count = len(r.split())
        if word_count >= 20:   score += 0.15
        elif word_count >= 10: score += 0.08

        # Crisis awareness
        if crisis_active:
            crisis_keywords = {
                "machine_breakdown": ["breakdown","machine","maintenance","reroute","repair","bearing","electrical"],
                "supply_shock":      ["supply","stock","supplier","order","shortage","material","backup"],
                "demand_spike":      ["demand","order","overtime","spike","urgent","capacity","accept"],
                "quality_failure":   ["quality","reject","rework","defect","maintenance","calibration","slow"],
                "worker_crisis":     ["worker","absent","contractor","staffing","shortage","overtime","personnel"],
            }
            matched = False
            for ct, words in crisis_keywords.items():
                if ct in crisis_type and any(w in r for w in words):
                    score += 0.25
                    matched = True
                    break
            if not matched and crisis_active:
                score -= 0.05  # crisis present but agent didn't mention it

        # References specific numbers
        numbers_found = re.findall(r'\b\d+\.?\d*\b', r)
        if len(numbers_found) >= 2:
            score += 0.15
        elif len(numbers_found) == 1:
            score += 0.05

        # References factory metrics
        metric_words = ["stock","units","cost","budget","worker","machine","target","capacity"]
        metrics_mentioned = sum(1 for w in metric_words if w in r)
        score += min(metrics_mentioned * 0.05, 0.20)

        # Severity awareness
        severity = situation.get("crisis_severity", "low").lower()
        if severity == "high":
            urgent_words = ["urgent","critical","immediately","emergency","severe","high priority"]
            if any(w in r for w in urgent_words):
                score += 0.10

        # Logical connectors (shows structured thinking)
        connectors = ["because","therefore","since","to prevent","in order to","due to","as a result"]
        if any(c in r for c in connectors):
            score += 0.05

        score = max(0.0, min(1.0, score))
        return ScoringResult(
            score, "keyword",
            f"keyword score: {score:.2f} (crisis_match={crisis_active}, words={word_count})"
        )

    def _format_situation(self, s: dict) -> str:
        crisis_line = ""
        if s.get("crisis_type") and "none" not in str(s.get("crisis_type","")).lower():
            crisis_line = (
                f"ACTIVE CRISIS: {s.get('crisis_message','unknown')}\n"
                f"Crisis severity: {s.get('crisis_severity','unknown')}\n"
                f"Available options: {s.get('resolution_options', [])}\n"
            )
        return (
            f"{crisis_line}"
            f"Stock remaining: {s.get('stock_days',0):.1f} days\n"
            f"Workers present: {s.get('workers_present',0)}/{s.get('workers_total',0)}\n"
            f"Units produced: {s.get('units_produced',0)}/{s.get('units_target',0)}\n"
            f"Cost today: Rs{s.get('cost_today',0):,} / Rs{s.get('budget_today',0):,}\n"
            f"Machines online: {s.get('machines_online',0)}/{s.get('machines_total',0)}"
        )
