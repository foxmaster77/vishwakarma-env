"""
inference.py  —  Vishwakarma Factory Environment Agent
=======================================================
Runs one full episode (16 steps) of the Vishwakarma factory RL environment.
The agent uses an LLM (via OpenAI-compatible API) to make factory decisions.

Environment variables (set these in your HF Space secrets):
  API_BASE_URL   — inference endpoint  (default provided for convenience)
  MODEL_NAME     — model to call       (default provided for convenience)
  HF_TOKEN       — your HuggingFace token  ← NO default, must be set externally
  LOCAL_IMAGE_NAME — optional, only needed if using from_docker_image()
"""

import os
import json
import time
import requests

from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Environment variables  (checklist items 2 & 3)
#     API_BASE_URL and MODEL_NAME have defaults.
#     HF_TOKEN deliberately has NO default.
# ─────────────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")                    # NO default — checklist item 3

# Optional — only used if you call from_docker_image()
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

# URL of your deployed Vishwakarma HF Space (the environment server)
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://tiny2520tots-vishwakarma-env.hf.space")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  OpenAI client configured via the env vars above  (checklist item 4)
# ─────────────────────────────────────────────────────────────────────────────

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Prompt builder — converts raw observation dict → LLM prompt
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(obs: dict) -> str:
    machines       = obs.get("machines", [])
    machines_online = sum(1 for m in machines if m.get("online", False))
    machines_total  = len(machines)

    crisis_section = ""
    alerts = obs.get("active_alerts", [])
    if alerts:
        alert = alerts[0]
        options = " | ".join(alert.get("resolution_options", []))
        crisis_section = (
            f"\n⚠ ACTIVE CRISIS [{alert.get('severity','').upper()}]:\n"
            f"  {alert.get('message','')}\n"
            f"  Options: {options}\n"
        )

    return f"""You are managing a factory. Make ONE decision for this time step.

FACTORY STATUS (Step {obs.get('step', 0)}/16):
  Machines: {machines_online}/{machines_total} online
  Workers:  {obs.get('workers_present', 0)}/{obs.get('workers_total', 0)}
  Stock:    {obs.get('stock_tons', 0):.1f}t ({obs.get('stock_days_remaining', 0):.1f} days remaining)
  Units:    {obs.get('units_produced_today', 0)}/{obs.get('units_target_today', 0)} produced today
  Cost:     Rs{obs.get('cost_today_INR', 0):,} / Rs{obs.get('budget_today_INR', 0):,}
  Reward:   {obs.get('reward', 0):.2f}  |  Feedback: {obs.get('feedback', '')}
{crisis_section}
AVAILABLE DIRECTIVES:
  run_normal, call_maintenance, order_stock, authorize_overtime,
  call_contractor, accept_order, decline_order, reroute_jobs, adjust_rate

Respond with JSON only — no markdown, no extra text:
{{
  "directive": "<directive>",
  "call_maintenance": <true/false>,
  "order_stock_tons": <0-20>,
  "order_stock_supplier": "<primary|backup>",
  "authorize_overtime_workers": <0-20>,
  "call_contractors": <0-10>,
  "adjust_production_rate": <0.1-1.5>,
  "accept_emergency_order": <true/false>,
  "reroute_from": "<machine name or null>",
  "reroute_to": "<machine name or null>",
  "reasoning": "<explain your decision in 1-2 sentences>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# 4.  LLM decision — calls model and parses JSON action
# ─────────────────────────────────────────────────────────────────────────────

def get_action(obs: dict) -> dict:
    """Ask the LLM what to do given the current observation."""
    prompt = build_prompt(obs)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.3,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        action = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: safe default action
        action = {
            "directive": "run_normal",
            "call_maintenance": False,
            "order_stock_tons": 0,
            "order_stock_supplier": "primary",
            "authorize_overtime_workers": 0,
            "call_contractors": 0,
            "adjust_production_rate": 1.0,
            "accept_emergency_order": False,
            "reroute_from": None,
            "reroute_to": None,
            "reasoning": "Could not parse LLM response, defaulting to run_normal.",
        }

    return action


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Environment HTTP calls
# ─────────────────────────────────────────────────────────────────────────────

def env_reset() -> dict:
    r = requests.post(f"{ENV_BASE_URL}/reset", timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(action: dict) -> dict:
    r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main loop  (checklist item 5 — START / STEP / END stdout format)
# ─────────────────────────────────────────────────────────────────────────────

def run():
    # ── START ─────────────────────────────────────────────────────────────────
    print("START")

    total_reward   = 0.0
    crises_resolved = 0

    # Reset the environment
    obs = env_reset()
    print(f"[INFO] Episode started — factory: {obs.get('shift', 1)} | "
          f"target: {obs.get('units_target_today', 0)} units/day")

    done = False
    step_num = 0

    while not done:
        # ── STEP ───────────────────────────────────────────────────────────────
        print(f"STEP {step_num}")

        # Get LLM decision
        action = get_action(obs)

        reasoning = action.get("reasoning", "")
        directive = action.get("directive", "run_normal")
        print(f"[STEP {step_num}] directive={directive} | reasoning={reasoning}")

        # Send action to environment
        obs = env_step(action)

        reward   = obs.get("reward", 0.0)
        feedback = obs.get("feedback", "")
        done     = obs.get("done", False)

        total_reward += reward

        # Track crises resolved (reward_breakdown has crisis_response > 0 when resolved)
        rb = obs.get("reward_breakdown", {})
        if rb.get("crisis_response", 0) > 0:
            crises_resolved += 1

        print(f"[STEP {step_num}] reward={reward:.2f} | cumulative={total_reward:.2f} | {feedback}")

        step_num += 1
        time.sleep(0.1)   # small pause — avoids hammering the server

    # ── END ───────────────────────────────────────────────────────────────────
    print("END")

    print(f"\n[RESULT] Episode complete in {step_num} steps")
    print(f"[RESULT] Total reward:      {total_reward:.2f}")
    print(f"[RESULT] Crises resolved:   {crises_resolved}")
    print(f"[RESULT] Units produced:    {obs.get('units_produced_today', 0)} "
          f"/ {obs.get('units_target_today', 0)}")


if __name__ == "__main__":
    run()
