"""
examples/demo_crisis_day.py

THE HACKATHON DEMO SCRIPT.
Run this on stage in Bangalore. Shows a factory under crisis,
with the agent responding step by step with live terminal output.

Usage:
    python examples/demo_crisis_day.py
    python examples/demo_crisis_day.py --factory pharma_packaging_hyderabad
    python examples/demo_crisis_day.py --factory textile_mill_surat
"""

import sys
import time
import argparse
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from vishwakarma_env.server.vishwakarma_environment import VishwakarmaEnvironment
from vishwakarma_env.models import VishwakarmaAction


# ─────────────────────────────────────────────
# Terminal colors
# ─────────────────────────────────────────────
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"


def bar(value, maximum, width=20, color=GREEN) -> str:
    filled = int((value / max(maximum, 1)) * width)
    b = "█" * filled + "░" * (width - filled)
    return f"{color}{b}{RESET}"


def print_header(factory_name: str):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  VISHWAKARMA FACTORY ENVIRONMENT{RESET}")
    print(f"{BOLD}  {factory_name}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")


def print_dashboard(obs, cumulative_reward: float):
    machines_online = sum(1 for m in obs.machines if m.online)
    machines_total  = len(obs.machines)
    machine_color   = GREEN if machines_online == machines_total else YELLOW if machines_online >= machines_total * 0.7 else RED

    prod_pct = obs.units_produced_today / max(obs.units_target_today, 1)
    prod_color = GREEN if prod_pct >= 0.9 else YELLOW if prod_pct >= 0.6 else RED

    cost_pct = obs.cost_today_INR / max(obs.budget_today_INR, 1)
    cost_color = GREEN if cost_pct <= 0.9 else YELLOW if cost_pct <= 1.0 else RED

    stock_color = GREEN if obs.stock_days_remaining >= 2 else YELLOW if obs.stock_days_remaining >= 1 else RED

    shift_label = f"Shift {obs.shift}"
    step_label  = f"Step {obs.step:02d}/16"

    print(f"\n{BOLD}── {shift_label} · {step_label} {'─'*30}{RESET}")
    print(
        f"  MACHINES   {machine_color}{bar(machines_online, machines_total, 12, machine_color)}{RESET}"
        f"  {machine_color}{machines_online}/{machines_total} online{RESET}"
    )
    print(
        f"  WORKERS    {bar(obs.workers_present, obs.workers_total, 12)}"
        f"  {obs.workers_present}/{obs.workers_total}"
    )
    print(
        f"  STOCK      {stock_color}{bar(obs.stock_days_remaining, 5, 12, stock_color)}{RESET}"
        f"  {stock_color}{obs.stock_tons:.1f}t ({obs.stock_days_remaining:.1f} days){RESET}"
    )
    print(
        f"  PRODUCTION {prod_color}{bar(obs.units_produced_today, obs.units_target_today, 12, prod_color)}{RESET}"
        f"  {prod_color}{obs.units_produced_today}/{obs.units_target_today} units{RESET}"
    )
    print(
        f"  COST       {cost_color}{bar(obs.cost_today_INR, obs.budget_today_INR, 12, cost_color)}{RESET}"
        f"  {cost_color}₹{obs.cost_today_INR:,} / ₹{obs.budget_today_INR:,}{RESET}"
    )

    if obs.active_alerts:
        for alert in obs.active_alerts:
            sev_color = RED if alert.severity == "high" else YELLOW if alert.severity == "medium" else CYAN
            print(f"\n  {sev_color}⚠  CRISIS [{alert.severity.upper()}]: {alert.message}{RESET}")
            print(f"     Options: {', '.join(alert.resolution_options)}")

    rb = obs.reward_breakdown
    if rb:
        breakdown = (
            f"prod={rb.get('production',0):+.2f} "
            f"cost={rb.get('cost',0):+.2f} "
            f"crisis={rb.get('crisis',0):+.2f} "
            f"safety={rb.get('safety',0):+.2f} "
            f"lt={rb.get('long_term',0):+.2f}"
        )
        reward_color = GREEN if obs.reward >= 0.5 else YELLOW if obs.reward >= 0 else RED
        print(f"\n  REWARD  {reward_color}{obs.reward:+.4f}{RESET}  {DIM}[{breakdown}]{RESET}")
        print(f"  TOTAL   {BOLD}{cumulative_reward:+.4f}{RESET}")


def agent_decide(obs) -> VishwakarmaAction:
    """
    Rule-based demo agent that responds to crises intelligently.
    In the real hackathon, this is replaced by an LLM.
    """
    # No crisis — run normally, but maintain buffer stock
    if not obs.active_alerts:
        # Check if we should order stock proactively
        if obs.stock_days_remaining < 1.5:
            return VishwakarmaAction(
                directive="order_stock",
                order_stock_tons=2.0,
                order_stock_supplier="primary",
                reasoning="Stock below 1.5 days — proactive reorder before shortage"
            )
        # Production behind — authorize overtime
        if obs.units_produced_today < obs.units_target_today * 0.6 and obs.step > 8:
            return VishwakarmaAction(
                directive="authorize_overtime",
                authorize_overtime_workers=6,
                reasoning="Behind target at shift 2 midpoint — overtime needed"
            )
        return VishwakarmaAction(
            directive="run_normal",
            reasoning="All systems normal. Maintaining standard operations."
        )

    # There's a crisis — respond based on type
    alert = obs.active_alerts[0]
    crisis = alert.crisis_type

    if crisis == "machine_breakdown":
        # Find offline machine
        offline = [m.name for m in obs.machines if not m.online]
        online  = [m.name for m in obs.machines if m.online]
        reroute_target = online[0] if online else None
        return VishwakarmaAction(
            directive="reroute_jobs",
            reroute_from=offline[0] if offline else None,
            reroute_to=reroute_target,
            call_maintenance=True,
            reasoning=f"Machine down. Rerouting to {reroute_target} and calling maintenance."
        )

    elif crisis == "supply_shock":
        return VishwakarmaAction(
            directive="order_stock",
            order_stock_tons=3.0,
            order_stock_supplier="backup",
            reasoning="Primary supplier delayed. Emergency order from backup (Mumbai). Premium cost justified."
        )

    elif crisis == "demand_spike":
        return VishwakarmaAction(
            directive="accept_order",
            authorize_overtime_workers=8,
            accept_emergency_order=True,
            reasoning="Revenue opportunity. Accepting order with overtime. Margin covers premium."
        )

    elif crisis == "quality_failure":
        return VishwakarmaAction(
            directive="reroute_jobs",
            call_maintenance=True,
            reasoning="Quality failure likely due to tool wear. Halting batch, calling QC and maintenance."
        )

    elif crisis == "worker_crisis":
        return VishwakarmaAction(
            directive="call_contractor",
            call_contractors=3,
            reasoning="Workers absent. Hiring contractors to maintain minimum staffing."
        )

    return VishwakarmaAction(
        directive="run_normal",
        reasoning="Monitoring situation."
    )


def run_demo(factory_id: str = "auto_components_pune", delay: float = 0.8):
    env = VishwakarmaEnvironment(factory_id=factory_id, seed=42)
    obs = env.reset()

    print_header(obs.feedback.split(".")[0] if obs.feedback else factory_id)
    print(f"{DIM}{obs.feedback[:200]}...{RESET}")

    cumulative_reward = 0.0

    while not obs.done:
        time.sleep(delay)

        # Agent decides
        action = agent_decide(obs)

        # Print what agent is doing
        print(f"\n  {BOLD}AGENT → {action.directive.upper()}{RESET}")
        if action.reasoning:
            print(f"  {DIM}Reasoning: {action.reasoning}{RESET}")

        # Step environment
        obs = env.step(action)
        cumulative_reward += obs.reward

        # Print dashboard
        print_dashboard(obs, cumulative_reward)

        # Print any important feedback lines
        for line in obs.feedback.split("\n"):
            line = line.strip()
            if line and not line.startswith("Step "):
                if "CRISIS" in line or "⚠" in line:
                    print(f"  {RED}{line}{RESET}")
                elif "✓" in line or "back online" in line:
                    print(f"  {GREEN}{line}{RESET}")
                elif "BOTTLENECK" in line:
                    print(f"  {YELLOW}{line}{RESET}")

    # Final summary
    state = env.state_info()
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN}  EPISODE COMPLETE{RESET}")
    print(f"{'='*60}")
    print(f"  Units produced  : {state['units_produced']:,} / {state['units_target']:,}")
    hit = state['units_produced'] >= state['units_target']
    print(f"  Target met      : {GREEN+'YES ✓'+RESET if hit else RED+'NO ✗'+RESET}")
    print(f"  Crises resolved : {state['crises_resolved']}/{state['crises_total']}")
    print(f"  Safety incidents: {state['safety_incidents']}")
    print(f"  Total reward    : {BOLD}{state['cumulative_reward']:.4f}{RESET}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vishwakarma Factory Demo")
    parser.add_argument("--factory", default="auto_components_pune",
                        choices=["auto_components_pune",
                                 "pharma_packaging_hyderabad",
                                 "textile_mill_surat"])
    parser.add_argument("--delay", type=float, default=0.8,
                        help="Seconds between steps (default 0.8)")
    args = parser.parse_args()
    run_demo(factory_id=args.factory, delay=args.delay)
