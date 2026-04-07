---
title: Vishwakarma Factory Environment
emoji: 🏭
colorFrom: yellow
colorTo: red
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - factory
  - manufacturing
  - india
  - grpo
license: mit
---

# 🏭 vishwakarma-env

> *Vishwakarma — the Hindu god of architecture, engineering, and craft.*

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible RL environment for **AI-driven factory operations**.

Train LLMs to manage Indian manufacturing plants — responding to machine breakdowns, supply shocks, demand spikes, quality failures, and worker crises across a simulated work day.

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Space secrets

Set these in **Settings → Variables and secrets** before deploying:

| Secret | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | Your HuggingFace write token (used as API key for the inference endpoint) |
| `API_BASE_URL` | Yes | Inference endpoint base URL, e.g. `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | Yes | Model name as the endpoint knows it, e.g. `Qwen/Qwen2.5-72B-Instruct` |
| `FACTORY_ID` | No | Factory to use (default: `auto_components_pune`) |

When `API_BASE_URL` and `MODEL_NAME` are set, the Space automatically runs a 3-episode inference demo on startup using `inference.py`.

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

The agent that learns to **think ahead** — maintaining buffer stock, building supplier diversity, not burning out workers with overtime — scores highest.

---

## Quick start

```bash
pip install -e .
python examples/demo_crisis_day.py
python examples/demo_crisis_day.py --factory pharma_packaging_hyderabad
```

---

## inference.py usage

```bash
# Rule-based baseline (no model needed)
python inference.py --agent mock --episodes 5 --verbose

# Any OpenAI-compatible endpoint (vLLM, Ollama, Together, HF Inference)
python inference.py \
    --agent openai \
    --base-url  $API_BASE_URL \
    --model     $MODEL_NAME \
    --api-key   $HF_TOKEN \
    --episodes  5 --verbose

# Online GRPO training loop: rollouts → update weights → repeat
python inference.py \
    --agent openai \
    --base-url  $API_BASE_URL \
    --model     $MODEL_NAME \
    --api-key   $HF_TOKEN \
    --train --trainer vllm --trainer-url $API_BASE_URL \
    --episodes 8 --train-rounds 5
```

---

## Python client

```python
from vishwakarma_env import VishwakarmaEnv, VishwakarmaAction

async with VishwakarmaEnv(base_url="https://YOUR-USERNAME-vishwakarma-env.hf.space") as env:
    obs = await env.reset()
    while not obs.done:
        action = VishwakarmaAction(
            directive="run_normal",
            reasoning="All systems nominal."
        )
        obs = await env.step(action)
        print(f"Step {obs.step}: reward={obs.reward:.4f}")
```

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

---

## Factory configurations

| Factory | City | Industry | Target/day | Budget/day |
|---------|------|----------|-----------|-----------|
| `auto_components_pune` | Pune | Auto components | 600 units | ₹2,80,000 |
| `pharma_packaging_hyderabad` | Hyderabad | Pharma packaging | 4000 units | ₹1,80,000 |
| `textile_mill_surat` | Surat | Textile machinery | 300 units | ₹3,50,000 |

---

## Architecture

```
vishwakarma_env/
├── models.py                        # VishwakarmaAction, Observation, State
├── client.py                        # OpenEnv HTTP client
├── server/
│   ├── vishwakarma_environment.py   # Episode orchestrator
│   ├── crisis_engine.py             # 5 crisis types, probabilistic
│   ├── production_simulator.py      # M/M/c queuing theory throughput
│   └── reward_engine.py             # 5-component reward function
├── data/
│   ├── machines.json                # 8 machine types with specs
│   ├── suppliers.json               # 4 supplier profiles
│   └── factories.json               # 3 factory configurations
├── inference.py                     # Run / train agents (mock, HF, OpenAI, Claude)
└── examples/
    ├── demo_crisis_day.py           # Live terminal demo
    └── grpo_training.py             # TRL/GRPO training integration
```

---

## Server API

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Episode metadata |
| `GET` | `/health` | Health check |
| `GET` | `/factories` | List factory configs |
| `POST` | `/multi/reset` | Start 2-factory episode |
| `POST` | `/multi/step` | Step 2-factory episode |

---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
