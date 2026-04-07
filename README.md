# 🏭 vishwakarma-env

> *Vishwakarma — the Hindu god of architecture, engineering, and craft.*

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible RL environment for **AI-driven factory operations**.

Train LLMs to manage Indian manufacturing plants — responding to machine breakdowns, supply shocks, demand spikes, quality failures, and worker crises across a simulated work day.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What this environment does

The agent operates a **running factory** — not designs one. Each episode is one factory work day (2 shifts, 16 steps). At any step, the environment can throw a real industrial crisis:

| Crisis | Example | Agent must... |
|--------|---------|--------------|
| Machine breakdown | CNC mill bearing failure | Reroute jobs, call maintenance |
| Supply shock | Steel supplier delayed on NH-48 | Emergency order from backup supplier |
| Demand spike | Urgent Maruti KANBAN request | Accept with overtime or decline |
| Quality failure | 15% reject rate, tool wear | Rework batch or scrap and restart |
| Worker crisis | 4 absent due to food poisoning | Hire contractors or redistribute tasks |

The agent that learns to **think ahead** — maintaining buffer stock, building supplier diversity, not burning out workers with overtime — scores highest. The reward function punishes short-termism.

---

## Quick start

```bash
# Install
pip install -e .

# Run the live demo (no server needed)
python examples/demo_crisis_day.py

# Try a different factory
python examples/demo_crisis_day.py --factory pharma_packaging_hyderabad
python examples/demo_crisis_day.py --factory textile_mill_surat
```

---

## Use with OpenEnv client

```python
from vishwakarma_env import VishwakarmaEnv, VishwakarmaAction

# Async (recommended for RL training loops)
async with VishwakarmaEnv(base_url="https://your-space.hf.space") as env:
    obs = await env.reset()
    while not obs.done:
        action = VishwakarmaAction(
            directive="run_normal",
            reasoning="All systems nominal."
        )
        obs = await env.step(action)
        print(f"Step {obs.step}: reward={obs.reward:.4f} | units={obs.units_produced_today}/{obs.units_target_today}")

# Sync
with VishwakarmaEnv(base_url="...").sync() as env:
    obs = env.reset()
    obs = env.step(VishwakarmaAction(directive="run_normal"))
```

---

## Available actions

```python
VishwakarmaAction(
    directive="run_normal",              # ProductionDirective value
    reroute_from="cnc_mill",             # machine to reroute FROM (optional)
    reroute_to="lathe",                  # machine to reroute TO (optional)
    call_maintenance=True,               # dispatch maintenance team
    order_stock_tons=2.0,               # emergency stock order qty
    order_stock_supplier="backup",       # "primary" or "backup"
    authorize_overtime_workers=6,        # workers for overtime shift
    call_contractors=3,                  # external contractors to hire
    adjust_production_rate=0.8,          # throttle production rate
    accept_emergency_order=True,         # accept a demand spike order
    reasoning="Explaining my decision"   # CoT — scored by LLM evaluator
)
```

Directives: `run_normal`, `reroute_jobs`, `call_maintenance`, `order_stock`,
`authorize_overtime`, `call_contractor`, `accept_order`, `decline_order`,
`adjust_rate`, `finalize_shift`

---

## Reward function (5 components)

```
Total reward = production(40%) + cost(25%) + crisis_response(20%) + safety(10%) + long_term(5%)
```

| Component | What it measures |
|-----------|-----------------|
| **Production** | Units produced vs daily target |
| **Cost** | Actual spend vs budget |
| **Crisis response** | Speed of response × damage contained |
| **Safety** | BIS IS-3696 compliance, zero incidents |
| **Long-term** | Buffer stock ≥ 2 days, supplier diversity ≥ 2 |

The long-term component is what makes this hard — it rewards the agent for *preventing* crises before they happen.

---

## Factory configurations

| Factory | City | Industry | Target/day | Budget/day |
|---------|------|----------|-----------|-----------|
| `auto_components_pune` | Pune | Auto components | 600 units | ₹2,80,000 |
| `pharma_packaging_hyderabad` | Hyderabad | Pharma packaging | 4000 units | ₹1,80,000 |
| `textile_mill_surat` | Surat | Textile machinery | 300 units | ₹3,50,000 |

All factories use **BIS (Bureau of Indian Standards)** safety codes and INR pricing.

---

## Architecture

```
vishwakarma_env/
├── models.py                   # VishwakarmaAction, Observation, State
├── client.py                   # OpenEnv HTTP client
├── server/
│   ├── vishwakarma_environment.py   # Episode orchestrator
│   ├── crisis_engine.py             # 5 crisis types, probabilistic
│   ├── production_simulator.py      # M/M/c queuing theory throughput
│   └── reward_engine.py             # 5-component reward function
├── data/
│   ├── machines.json                # 8 machine types with specs
│   ├── suppliers.json               # 4 supplier profiles
│   └── factories.json               # 3 factory configurations
└── examples/
    └── demo_crisis_day.py           # Live terminal demo
```

### Physics

Production throughput uses **M/M/c queuing theory** (operations research):
- `ρ = λ / (c × μ)` — utilization per station
- When `ρ ≥ 1.0` → bottleneck detected → degraded throughput
- Station results fed into bottleneck analysis with agent feedback

Safety uses **BIS IS-3696** minimum clearance standards (0.9m walkway, hazard staffing ratios).

---

## Running the server (Docker)

```bash
# Build
docker build -t vishwakarma-env .

# Run (default: Pune auto factory)
docker run -p 7860:7860 vishwakarma-env

# Different factory
docker run -p 7860:7860 -e FACTORY_ID=pharma_packaging_hyderabad vishwakarma-env
```

Server endpoints:
- `POST /reset` — start new episode
- `POST /step` — execute action
- `GET /state` — episode metadata
- `GET /health` — health check

---

## Running tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

11 tests covering reset, step, full episode, production, costs, crisis, rewards.

---

## Integration with TRL/GRPO

```python
import asyncio
from vishwakarma_env import VishwakarmaEnv, VishwakarmaAction

async def run_episode(model, base_url):
    rewards = []
    async with VishwakarmaEnv(base_url=base_url) as env:
        obs = await env.reset()
        while not obs.done:
            # Format observation as prompt for LLM
            prompt = format_observation_as_prompt(obs)
            # Get action from model
            response = model.generate(prompt)
            action = parse_action_from_response(response)
            obs = await env.step(action)
            rewards.append(obs.reward)
    return rewards
```

See `examples/grpo_training.py` for full TRL integration.

---

## Why this environment matters

Samsung, Tata, Maruti, and hundreds of Indian manufacturers are actively building AI systems to operate factories. The missing piece is **open RL training infrastructure** — environments that can train agents to handle real industrial complexity.

`vishwakarma-env` is that infrastructure. Built on OpenEnv. Open source. Ready to train on.

---

## License

MIT — use freely, contribute back.

## Acknowledgements

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) by Meta & Hugging Face.
Physics based on M/M/c queuing theory (Erlang, 1909).
Safety standards from BIS IS-3696.
