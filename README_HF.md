---
title: Vishwakarma Factory Environment
emoji: 🏭
colorFrom: orange
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - factory
  - manufacturing
  - india
  - grpo
license: mit
---

# Vishwakarma Factory Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible RL environment
for AI-driven factory operations. Train LLMs to manage Indian manufacturing
plants — responding to machine breakdowns, supply shocks, demand spikes,
quality failures, and worker crises.

## Space secrets

Set these in **Settings → Variables and secrets** before deploying:

| Secret | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | Your HuggingFace write token (used as API key for the inference endpoint) |
| `API_BASE_URL` | Yes | Inference endpoint base URL, e.g. `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | Yes | Model name as the endpoint knows it, e.g. `Qwen/Qwen2.5-72B-Instruct` |
| `FACTORY_ID` | No | Factory to use (default: `auto_components_pune`). Options: `auto_components_pune`, `pharma_packaging_hyderabad`, `textile_mill_surat` |

When `API_BASE_URL` and `MODEL_NAME` are set, the Space automatically runs a
3-episode inference demo on startup and logs results to the Space logs.

## Environment API

The Space exposes the Vishwakarma RL environment as a REST API on port 7860:

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Start a new episode → `VishwakarmaObservation` |
| `POST` | `/step` | Execute one action → `VishwakarmaObservation` |
| `GET` | `/state` | Episode metadata (step, crises, rewards) |
| `GET` | `/health` | Health check |
| `GET` | `/factories` | List available factory configs |
| `POST` | `/multi/reset` | Start a 2-factory shared-supplier episode |
| `POST` | `/multi/step` | Step the 2-factory episode |

## Using inference.py

`inference.py` in the repo root supports three agent modes and two training backends:

```bash
# Run against this Space using an OpenAI-compatible endpoint
python inference.py \
    --agent openai \
    --base-url  $API_BASE_URL \
    --model     $MODEL_NAME \
    --api-key   $HF_TOKEN \
    --episodes  5 --verbose

# Online GRPO training loop (collect rollouts → update weights → repeat)
python inference.py \
    --agent openai \
    --base-url  $API_BASE_URL \
    --model     $MODEL_NAME \
    --api-key   $HF_TOKEN \
    --train --trainer vllm --trainer-url $API_BASE_URL \
    --episodes 8 --train-rounds 5

# Rule-based baseline (no model needed)
python inference.py --agent mock --episodes 10 --verbose
```

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

## Environment details

- **16 steps/episode** — 2 shifts × 8 steps, each step = 1 hour
- **5 crisis types** — machine breakdown, supply shock, demand spike, quality failure, worker crisis
- **5-component reward** — production (40%) + cost (25%) + crisis response (20%) + safety (10%) + long-term (5%)
- **3 factory configs** — Pune auto components, Hyderabad pharma packaging, Surat textile mill
- **Physics** — M/M/c queuing theory throughput, BIS IS-3696 safety compliance
- **Pricing** — full INR cost model (machines, workers, contractors, overtime, stock)

See the [GitHub repo](https://github.com/AryanLuharuwala/vishwakarma-env) for full documentation.
