"""
inference.py — Run, evaluate, and continually train an agent on Vishwakarma.

Agent modes:
  --agent mock       Rule-based agent (no model needed, good baseline)
  --agent hf         HuggingFace transformers model loaded in-process
  --agent openai     Any OpenAI-compatible API: vLLM, Ollama, LM Studio,
                     Together AI, OpenAI, etc.  Set --base-url + --model.
  --agent claude     Anthropic Claude API

Training modes:
  --train            Interleave rollouts with GRPO updates (online loop)
  --train-only       Train from an existing trajectory buffer, no new rollouts
  --trainer local    GRPO via in-process TRL (default, requires --agent hf)
  --trainer vllm     Push GRPO jobs to a vLLM training server over HTTP
                     (works with --agent openai or --agent hf)

Usage examples:
  # vLLM inference + vLLM GRPO training loop
  python inference.py --agent openai \\
      --base-url http://localhost:8000 --model Qwen/Qwen2.5-7B-Instruct \\
      --train --trainer vllm --episodes 8 --train-rounds 5

  # Any OpenAI-compatible endpoint for inference only
  python inference.py --agent openai \\
      --base-url http://localhost:11434/v1 --model llama3  # Ollama
  python inference.py --agent openai \\
      --base-url https://api.together.xyz/v1 --model mistralai/Mixtral-8x7B-Instruct-v0.1 \\
      --api-key $TOGETHER_API_KEY

  # Local HF model + in-process GRPO
  python inference.py --agent hf --model Qwen/Qwen2.5-0.5B-Instruct \\
      --train --episodes 8 --train-rounds 5

  # Train-only from a saved buffer
  python inference.py --agent hf --model Qwen/Qwen2.5-0.5B-Instruct \\
      --train-only --buffer trajectories.jsonl

Trajectory buffer:
  Every rollout appends step records to --buffer (default: trajectories.jsonl).
  Each line: {"prompt":"...","response":"...","episode_reward":42.3,...}
  The training loop uses only the top-K episodes by reward for each GRPO update.

vLLM trainer API expected endpoints (POST):
  POST /train/grpo    — submit a GRPO job
  GET  /train/status/{job_id} — poll job status
  See VLLMGRPOTrainer docstring for the full request/response schema.

Output:
  Per-step table, episode summary, aggregate stats.
  Use --json for machine-readable JSON lines per step.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder (shared across all agents)
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(obs) -> str:
    """Convert a VishwakarmaObservation to an LLM-friendly prompt string."""
    machines_online = sum(1 for m in obs.machines if m.online)
    machines_total  = len(obs.machines)

    # List broken machines explicitly
    broken = [m.name for m in obs.machines if not m.online]
    machine_detail = f"  Online: {machines_online}/{machines_total}"
    if broken:
        machine_detail += f"  |  OFFLINE: {', '.join(broken)}"

    crisis_section = ""
    if obs.active_alerts:
        alert = obs.active_alerts[0]
        crisis_section = (
            f"\n⚠ ACTIVE CRISIS [{alert.severity.upper()}]: {alert.crisis_type}\n"
            f"  {alert.message}\n"
            f"  Resolution options: {' | '.join(alert.resolution_options)}\n"
        )

    budget_pct = (
        100 * obs.cost_today_INR / obs.budget_today_INR
        if obs.budget_today_INR > 0 else 0
    )
    prod_pct = (
        100 * obs.units_produced_today / obs.units_target_today
        if obs.units_target_today > 0 else 0
    )

    return f"""You are a factory floor manager. Decide the best action for this time step.

═══════════════════════════════════════
FACTORY STATUS  (Step {obs.step}/16, Shift {obs.shift})
═══════════════════════════════════════
Machines : {machine_detail}
Workers  : {obs.workers_present}/{obs.workers_total} present
Stock    : {obs.stock_tons:.1f} t  ({obs.stock_days_remaining:.1f} days remaining)
Production: {obs.units_produced_today}/{obs.units_target_today} units  ({prod_pct:.0f}% of target)
  Rate this step: {obs.production_rate_per_hour:.0f} units/hr
Cost     : Rs{obs.cost_today_INR:,} / Rs{obs.budget_today_INR:,}  ({budget_pct:.0f}% of budget)
{crisis_section}
═══════════════════════════════════════
DIRECTIVES AVAILABLE:
  run_normal         — standard operation
  call_maintenance   — dispatch maintenance team (costs Rs4,500)
  reroute_jobs       — reroute work around a broken machine
  order_stock        — emergency stock order (primary cheaper, backup more reliable)
  authorize_overtime — keep workers after shift (Rs850/worker)
  call_contractor    — hire external workers (Rs1,200/person)
  accept_order       — accept emergency demand spike
  decline_order      — decline demand spike (no penalty)
  adjust_rate        — change production speed multiplier
  finalize_shift     — end current shift early
═══════════════════════════════════════

Respond with a JSON object only (no markdown, no extra text):
{{
  "directive": "<one of the directives above>",
  "call_maintenance": <true/false>,
  "order_stock_tons": <0.0 to 20.0>,
  "order_stock_supplier": "<primary or backup>",
  "authorize_overtime_workers": <0 to 20>,
  "call_contractors": <0 to 10>,
  "adjust_production_rate": <0.1 to 1.5>,
  "accept_emergency_order": <true/false>,
  "reroute_from": "<machine name or null>",
  "reroute_to": "<machine name or null>",
  "reasoning": "<1-2 sentences explaining your decision, reference specific numbers>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Action parser (shared)
# ─────────────────────────────────────────────────────────────────────────────

def parse_action(text: str):
    """Parse an LLM JSON response into VishwakarmaAction. Falls back to run_normal."""
    from vishwakarma_env.models import VishwakarmaAction

    clean = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start >= 0 and end > start:
            d = json.loads(clean[start:end])
            return VishwakarmaAction(
                directive                 = d.get("directive", "run_normal"),
                call_maintenance          = bool(d.get("call_maintenance", False)),
                order_stock_tons          = float(d.get("order_stock_tons", 0.0)),
                order_stock_supplier      = d.get("order_stock_supplier", "primary"),
                authorize_overtime_workers= int(d.get("authorize_overtime_workers", 0)),
                call_contractors          = int(d.get("call_contractors", 0)),
                adjust_production_rate    = float(d.get("adjust_production_rate", 1.0)),
                accept_emergency_order    = bool(d.get("accept_emergency_order", False)),
                reroute_from              = d.get("reroute_from") or None,
                reroute_to                = d.get("reroute_to") or None,
                reasoning                 = d.get("reasoning", ""),
            )
    except Exception:
        pass

    return __import__("vishwakarma_env.models", fromlist=["VishwakarmaAction"]).VishwakarmaAction(
        directive="run_normal",
        reasoning="Parse failed — defaulting to normal operations.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent implementations
# ─────────────────────────────────────────────────────────────────────────────

class MockAgent:
    """
    Rule-based agent. Responds deterministically to each crisis type.
    Useful as a baseline and for smoke-testing without any model.
    """
    name = "mock-rule-based"

    def act(self, obs) -> tuple:
        from vishwakarma_env.models import VishwakarmaAction
        if not obs.active_alerts:
            # No crisis: top up stock if running low, else run normally
            if obs.stock_days_remaining < 1.5:
                return VishwakarmaAction(
                    directive="order_stock",
                    order_stock_tons=2.0,
                    order_stock_supplier="primary",
                    reasoning=(
                        f"Buffer low ({obs.stock_days_remaining:.1f} days). "
                        "Ordering 2t from primary to rebuild safety stock."
                    ),
                ), None

            return VishwakarmaAction(
                directive="run_normal",
                reasoning="All systems nominal. Running at full capacity.",
            ), None

        alert = obs.active_alerts[0]
        ct = str(alert.crisis_type).upper()

        if "MACHINE" in ct:
            action = VishwakarmaAction(
                directive="call_maintenance",
                call_maintenance=True,
                reasoning=(
                    f"Machine breakdown detected (severity={alert.severity}). "
                    "Dispatching maintenance to restore online capacity."
                ),
            )
        elif "SUPPLY" in ct:
            tons = 3.0 if alert.severity in ("medium", "high") else 2.0
            action = VishwakarmaAction(
                directive="order_stock",
                order_stock_tons=tons,
                order_stock_supplier="backup",
                reasoning=(
                    f"Supply shock ({alert.severity}). Ordering {tons}t from backup "
                    "supplier to avoid production halt."
                ),
            )
        elif "DEMAND" in ct:
            ot = 6 if alert.severity == "high" else 4
            action = VishwakarmaAction(
                directive="accept_order",
                accept_emergency_order=True,
                authorize_overtime_workers=ot,
                reasoning=(
                    f"Demand spike ({alert.severity}). Accepting with {ot} overtime "
                    "workers — revenue likely exceeds extra labour cost."
                ),
            )
        elif "QUALITY" in ct:
            rate = 0.7 if alert.severity == "high" else 0.8
            action = VishwakarmaAction(
                directive="call_maintenance",
                call_maintenance=True,
                adjust_production_rate=rate,
                reasoning=(
                    f"Quality failure ({alert.severity}). Slowing to {int(rate*100)}% "
                    "and calling maintenance to fix root cause."
                ),
            )
        else:  # WORKER_CRISIS
            n = 5 if alert.severity == "high" else 3
            action = VishwakarmaAction(
                directive="call_contractor",
                call_contractors=n,
                reasoning=(
                    f"Worker crisis ({alert.severity}). Hiring {n} contractors "
                    "to maintain floor staffing levels."
                ),
            )

        return action, None


class HFAgent:
    """
    HuggingFace transformers-based agent.
    Loads any causal LM (e.g. Qwen, Llama, Mistral, or a fine-tuned checkpoint).
    """
    name: str

    def __init__(self, model_id: str, temperature: float = 0.7, max_new_tokens: int = 256):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            sys.exit("transformers not installed. Run: pip install transformers torch")

        print(f"Loading {model_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.temperature    = temperature
        self.max_new_tokens = max_new_tokens
        self.name           = f"hf:{model_id}"
        print(f"Model ready on {next(self.model.parameters()).device}")

    def act(self, obs) -> tuple:
        import torch
        prompt = build_prompt(obs)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens  = self.max_new_tokens,
                temperature     = self.temperature,
                do_sample       = self.temperature > 0,
                pad_token_id    = self.tokenizer.eos_token_id,
            )
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return parse_action(response), response


class ClaudeAgent:
    """
    Calls the Claude API (anthropic SDK) for each action.
    Requires ANTHROPIC_API_KEY env var or --api-key argument.
    """

    def __init__(self, model_id: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            sys.exit("anthropic SDK not installed. Run: pip install anthropic")

        self.client   = __import__("anthropic").Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model_id = model_id
        self.name     = f"claude:{model_id}"

    def act(self, obs) -> tuple:
        prompt = build_prompt(obs)
        msg = self.client.messages.create(
            model      = self.model_id,
            max_tokens = 512,
            messages   = [{"role": "user", "content": prompt}],
        )
        response = msg.content[0].text
        return parse_action(response), response


class OpenAICompatAgent:
    """
    Calls any OpenAI-compatible chat completions endpoint.

    Works with:
      - vLLM          python -m vllm.entrypoints.openai.api_server --model ...
      - Ollama        http://localhost:11434/v1
      - LM Studio     http://localhost:1234/v1
      - Together AI   https://api.together.xyz/v1
      - OpenAI        https://api.openai.com/v1
      - Any server that speaks POST /v1/chat/completions

    Args:
        base_url:    Server root URL.  "/v1" is appended if not already present.
        model_id:    Model name as the server knows it.
        api_key:     Bearer token (use "EMPTY" for unauthenticated local servers).
        temperature: Sampling temperature.
        max_tokens:  Max completion tokens.
        timeout:     HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "EMPTY",
        temperature: float = 0.7,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            sys.exit("openai package not installed. Run: pip install openai")

        # Normalise URL: strip trailing slash, append /v1 if missing
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = url + "/v1"

        self.client      = __import__("openai").OpenAI(
            base_url=url,
            api_key=api_key,
            timeout=timeout,
        )
        self.model_id    = model_id
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.name        = f"openai-compat:{model_id}@{base_url}"

    def act(self, obs) -> tuple:
        prompt = build_prompt(obs)
        resp   = self.client.chat.completions.create(
            model       = self.model_id,
            messages    = [{"role": "user", "content": prompt}],
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
        )
        response = resp.choices[0].message.content or ""
        return parse_action(response), response


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory buffer — persists (prompt, response, reward) tuples to JSONL
# ─────────────────────────────────────────────────────────────────────────────

class TrajectoryBuffer:
    """
    Append-only JSONL store.  Each record = one environment step.
    At training time we group by episode and rank by total episode reward.
    """

    def __init__(self, path: str):
        self.path = path

    def append(self, record: dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def top_k_episodes(self, k: int) -> list[dict]:
        """
        Return individual step records belonging to the top-k episodes
        ranked by episode_reward. Used to build the GRPO training dataset.
        """
        records = self.load()
        if not records:
            return []

        # Group by (factory, seed) — unique episode identifier
        episodes: dict[tuple, list[dict]] = {}
        for r in records:
            key = (r.get("factory", ""), r.get("seed", 0))
            episodes.setdefault(key, []).append(r)

        # Sort episodes by episode_reward of their first record (all steps share it)
        ranked = sorted(
            episodes.values(),
            key=lambda steps: steps[0].get("episode_reward", 0.0),
            reverse=True,
        )

        # Flatten top-k episodes back to a step list
        flat = []
        for ep_steps in ranked[:k]:
            flat.extend(ep_steps)
        return flat

    def size(self) -> int:
        return len(self.load())


# ─────────────────────────────────────────────────────────────────────────────
# GRPO training loop (requires: trl, transformers, torch, datasets)
# ─────────────────────────────────────────────────────────────────────────────

def run_grpo_training(
    hf_agent: "HFAgent",
    buffer: TrajectoryBuffer,
    factory_id: str,
    checkpoint_dir: str,
    round_idx: int,
    top_k: int,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    epochs: int,
):
    """
    Run one round of GRPO fine-tuning on the top-k episodes stored in buffer.

    Strategy: treat each step's (prompt → response) as a training sample,
    using the *episode* total reward as the scalar reward signal. This is
    equivalent to the standard GRPO formulation where the reward is assigned
    uniformly across the trajectory.
    """
    try:
        from trl import GRPOConfig, GRPOTrainer
        from datasets import Dataset
    except ImportError:
        print("  [train] trl / datasets not installed. Run: pip install trl datasets")
        print("  [train] Skipping GRPO update for this round.")
        return

    steps = buffer.top_k_episodes(top_k)
    if not steps:
        print("  [train] Buffer is empty — skipping training round.")
        return

    # Only use steps that have a real LLM response (not mock/fallback)
    trainable = [s for s in steps if s.get("response", "").strip()]
    if not trainable:
        print("  [train] No LLM responses in buffer (mock agent?). Skipping.")
        return

    print(f"\n  [train] Round {round_idx} — {len(trainable)} steps from top-{top_k} episodes")
    print(f"  [train] Reward range: "
          f"{min(s['episode_reward'] for s in trainable):.1f} … "
          f"{max(s['episode_reward'] for s in trainable):.1f}")

    # Build dataset: prompt + expected completion (the stored LLM response)
    dataset = Dataset.from_list([
        {"prompt": s["prompt"], "completion": s["response"]}
        for s in trainable
    ])

    # Reward function: return the stored episode_reward for each completion.
    # GRPO will compute advantages relative to the mean within each batch.
    reward_lookup = {s["prompt"]: s["episode_reward"] for s in trainable}

    def reward_fn(completions, prompts, **kwargs):
        return [float(reward_lookup.get(p, 0.0)) for p in prompts]

    out_dir = os.path.join(checkpoint_dir, f"round-{round_idx}")
    config = GRPOConfig(
        output_dir                  = out_dir,
        num_train_epochs            = epochs,
        per_device_train_batch_size = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = learning_rate,
        logging_steps               = max(1, len(trainable) // (batch_size * 4)),
        save_steps                  = max(10, len(trainable) // batch_size),
        max_completion_length       = 256,
        num_generations             = min(4, batch_size),
        report_to                   = "none",
    )

    trainer = GRPOTrainer(
        model         = hf_agent.model,
        tokenizer     = hf_agent.tokenizer,
        reward_funcs  = reward_fn,
        args          = config,
        train_dataset = dataset,
    )

    print(f"  [train] Starting GRPO update …")
    trainer.train()
    trainer.save_model(out_dir)
    print(f"  [train] Checkpoint saved → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# vLLM GRPO trainer — submits training jobs to a vLLM training server over HTTP
# ─────────────────────────────────────────────────────────────────────────────

class VLLMGRPOTrainer:
    """
    Sends GRPO training jobs to a vLLM server that exposes a training API.

    Expected server endpoints
    ─────────────────────────
    POST /train/grpo
      Request body:
        {
          "model": "<model_id>",
          "training_data": [
            {"prompt": "...", "response": "...", "reward": 42.3},
            ...
          ],
          "config": {
            "learning_rate": 1e-5,
            "epochs": 1,
            "batch_size": 4,
            "num_generations": 4
          }
        }
      Response:
        {"job_id": "grpo-abc123", "status": "queued"}

    GET /train/status/{job_id}
      Response:
        {"job_id": "...", "status": "running|completed|failed",
         "progress": 0.65, "checkpoint": "/path/to/ckpt"}

    POST /train/cancel/{job_id}   (optional — used on KeyboardInterrupt)

    If your vLLM build doesn't have a training API yet, start it with:
      python -m vllm.entrypoints.openai.api_server \\
          --model Qwen/Qwen2.5-7B-Instruct --enable-lora
    and point --trainer-url at the same host — the trainer will log a clear
    error and fall back gracefully rather than crashing the rollout loop.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: str = "EMPTY",
        poll_interval: float = 5.0,
        timeout: float = 30.0,
    ):
        import urllib.request as _ur
        self._ur          = _ur
        url               = base_url.rstrip("/")
        self.base_url     = url
        self.model_id     = model_id
        self.api_key      = api_key
        self.poll_interval= poll_interval
        self.timeout      = timeout

    def _post(self, path: str, body: dict) -> dict:
        import urllib.request, urllib.error
        data    = json.dumps(body).encode()
        req     = urllib.request.Request(
            f"{self.base_url}{path}",
            data    = data,
            headers = {
                "Content-Type" : "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method  = "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"HTTP {e.code} from {path}: {body}") from e

    def _get(self, path: str) -> dict:
        import urllib.request, urllib.error
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} from {path}") from e

    def train(
        self,
        buffer: "TrajectoryBuffer",
        round_idx: int,
        top_k: int,
        batch_size: int,
        learning_rate: float,
        epochs: int,
        checkpoint_dir: str,
    ):
        steps = buffer.top_k_episodes(top_k)
        trainable = [s for s in steps if s.get("response", "").strip()]
        if not trainable:
            print("  [vllm-train] No LLM responses in buffer. Skipping.")
            return

        print(f"\n  [vllm-train] Round {round_idx} — submitting {len(trainable)} steps "
              f"from top-{top_k} episodes to {self.base_url}")

        training_data = [
            {
                "prompt"  : s["prompt"],
                "response": s["response"],
                "reward"  : s["episode_reward"],
            }
            for s in trainable
        ]

        payload = {
            "model"        : self.model_id,
            "training_data": training_data,
            "config": {
                "learning_rate"  : learning_rate,
                "epochs"         : epochs,
                "batch_size"     : batch_size,
                "num_generations": min(4, batch_size),
                "checkpoint_dir" : os.path.join(checkpoint_dir, f"round-{round_idx}"),
            },
        }

        try:
            resp = self._post("/train/grpo", payload)
        except RuntimeError as e:
            print(f"  [vllm-train] Could not submit job: {e}")
            print("  [vllm-train] Is the vLLM training API running? Skipping round.")
            return

        job_id = resp.get("job_id", "unknown")
        print(f"  [vllm-train] Job submitted: {job_id}  (status={resp.get('status')})")

        # Poll until done
        while True:
            time.sleep(self.poll_interval)
            try:
                status = self._get(f"/train/status/{job_id}")
            except RuntimeError as e:
                print(f"  [vllm-train] Status poll failed: {e}. Will retry.")
                continue

            pct  = status.get("progress", 0.0)
            st   = status.get("status", "unknown")
            ckpt = status.get("checkpoint", "")
            print(f"  [vllm-train] {job_id}  status={st}  progress={pct:.0%}  ckpt={ckpt}")

            if st == "completed":
                print(f"  [vllm-train] Training complete. Checkpoint: {ckpt}")
                break
            elif st == "failed":
                print(f"  [vllm-train] Training FAILED for job {job_id}.")
                break


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner (local environment)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    episode     : int
    factory_id  : str
    seed        : int
    total_reward: float
    units_produced: int
    units_target  : int
    production_pct: float
    cost_INR      : int
    budget_INR    : int
    cost_pct      : float
    crises_resolved: int
    crises_total   : int
    safety_incidents: int
    steps_completed : int
    elapsed_sec     : float


def run_episode(
    agent,
    factory_id: str,
    seed: int,
    episode_idx: int,
    verbose: bool,
    emit_json: bool,
    buffer: "TrajectoryBuffer | None" = None,
) -> EpisodeResult:
    from vishwakarma_env.server.vishwakarma_environment import VishwakarmaEnvironment

    env   = VishwakarmaEnvironment(factory_id=factory_id, seed=seed)
    obs   = env.reset()
    total = 0.0
    t0    = time.time()

    # Collect step records before we know the episode total reward.
    # We'll patch episode_reward in after the episode finishes.
    step_records: list[dict] = []

    if verbose and not emit_json:
        print(f"\n{'═'*72}")
        print(f"  Episode {episode_idx+1}  |  factory={factory_id}  |  seed={seed}")
        print(f"{'═'*72}")
        print(f"  {'Step':>4}  {'Directive':<22}  {'Reward':>7}  {'Cumul':>7}  {'Crisis'}")
        print(f"  {'-'*60}")

    for step in range(env.TOTAL_STEPS):
        prompt           = build_prompt(obs)
        action, raw_resp = agent.act(obs)
        obs              = env.step(action)
        total           += obs.reward

        if emit_json:
            print(json.dumps({
                "episode": episode_idx + 1,
                "step"   : obs.step,
                "directive": action.directive,
                "reward" : round(obs.reward, 4),
                "cumulative_reward": round(total, 4),
                "crisis" : obs.crisis_type,
                "crisis_severity": obs.crisis_severity,
                "units_produced": obs.units_produced_today,
                "units_target": obs.units_target_today,
                "reasoning": action.reasoning,
            }))
        elif verbose:
            crisis_tag = ""
            if obs.active_alerts:
                alert      = obs.active_alerts[0]
                crisis_tag = f"[{alert.severity.upper()}] {alert.crisis_type}"
            print(
                f"  {obs.step:>4}  {action.directive:<22}  "
                f"{obs.reward:>+7.3f}  {total:>+7.2f}  {crisis_tag}"
            )

        if buffer is not None:
            step_records.append({
                "prompt"        : prompt,
                "response"      : raw_resp or "",
                "step_reward"   : round(obs.reward, 4),
                "episode_reward": 0.0,   # filled in below
                "factory"       : factory_id,
                "seed"          : seed,
                "step"          : obs.step,
                "directive"     : action.directive,
            })

        if obs.done:
            break

    info    = env.state_info()
    elapsed = time.time() - t0

    # Patch episode_reward now that we know the total, then flush to disk
    if buffer is not None:
        for rec in step_records:
            rec["episode_reward"] = round(total, 4)
            buffer.append(rec)

    result = EpisodeResult(
        episode         = episode_idx + 1,
        factory_id      = factory_id,
        seed            = seed,
        total_reward    = round(total, 4),
        units_produced  = info["units_produced"],
        units_target    = info["units_target"],
        production_pct  = round(100 * info["units_produced"] / max(info["units_target"], 1), 1),
        cost_INR        = obs.cost_today_INR,
        budget_INR      = obs.budget_today_INR,
        cost_pct        = round(100 * obs.cost_today_INR / max(obs.budget_today_INR, 1), 1),
        crises_resolved = info["crises_resolved"],
        crises_total    = info["crises_total"],
        safety_incidents= info.get("safety_incidents", 0),
        steps_completed = obs.step,
        elapsed_sec     = round(elapsed, 2),
    )

    if not emit_json:
        _print_episode_summary(result, verbose)

    return result


def _print_episode_summary(r: EpisodeResult, verbose: bool):
    sep = "─" * 72 if verbose else ""
    if sep:
        print(f"  {sep}")
    print(
        f"  Ep{r.episode:02d} summary │ "
        f"reward={r.total_reward:+.2f} │ "
        f"units={r.units_produced}/{r.units_target} ({r.production_pct:.0f}%) │ "
        f"cost={r.cost_pct:.0f}% of budget │ "
        f"crises {r.crises_resolved}/{r.crises_total} resolved │ "
        f"safety={r.safety_incidents} │ "
        f"{r.elapsed_sec:.1f}s"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────────────────────

def print_aggregate(results: list[EpisodeResult]):
    n = len(results)
    if n == 0:
        return
    rewards  = [r.total_reward    for r in results]
    prod_pct = [r.production_pct  for r in results]
    cost_pct = [r.cost_pct        for r in results]
    cr_rate  = [
        r.crises_resolved / max(r.crises_total, 1) * 100
        for r in results
    ]

    def _avg(lst): return sum(lst) / len(lst)
    def _min(lst): return min(lst)
    def _max(lst): return max(lst)

    print(f"\n{'═'*72}")
    print(f"  AGGREGATE STATS  ({n} episode{'s' if n > 1 else ''})")
    print(f"{'═'*72}")
    print(f"  {'Metric':<30} {'Mean':>8}  {'Min':>8}  {'Max':>8}")
    print(f"  {'-'*54}")
    print(f"  {'Total reward':<30} {_avg(rewards):>+8.2f}  {_min(rewards):>+8.2f}  {_max(rewards):>+8.2f}")
    print(f"  {'Production % of target':<30} {_avg(prod_pct):>8.1f}  {_min(prod_pct):>8.1f}  {_max(prod_pct):>8.1f}")
    print(f"  {'Cost % of budget':<30} {_avg(cost_pct):>8.1f}  {_min(cost_pct):>8.1f}  {_max(cost_pct):>8.1f}")
    print(f"  {'Crisis resolution %':<30} {_avg(cr_rate):>8.1f}  {_min(cr_rate):>8.1f}  {_max(cr_rate):>8.1f}")
    print(f"{'═'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Remote (HTTP) runner — wraps VishwakarmaEnv client
# ─────────────────────────────────────────────────────────────────────────────

def run_remote(agent, base_url: str, factory_id: str, episodes: int, verbose: bool):
    """Run inference against a remote server (HuggingFace Space or local Docker)."""
    import asyncio
    from vishwakarma_env.client import VishwakarmaEnv

    async def _run():
        results = []
        async with VishwakarmaEnv(base_url=base_url) as env:
            for ep in range(episodes):
                obs   = await env.reset(factory_id=factory_id)
                total = 0.0
                print(f"\nEpisode {ep+1}: target={obs.units_target_today} units/day")
                for _ in range(16):
                    action, _ = agent.act(obs)
                    obs       = await env.step(action)
                    total    += obs.reward
                    if verbose:
                        print(
                            f"  Step {obs.step:02d}: {action.directive:<22} "
                            f"reward={obs.reward:+.3f}  total={total:+.2f}"
                        )
                    if obs.done:
                        break
                print(f"  Episode {ep+1} complete — total reward: {total:.2f}")
                results.append(total)
        print(f"\nAll episodes: {[round(r, 2) for r in results]}")
        print(f"Mean reward : {sum(results)/len(results):.2f}")

    asyncio.run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vishwakarma inference — run an agent on the factory environment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Agent selection
    parser.add_argument(
        "--agent", default="mock",
        choices=["mock", "hf", "openai", "claude"],
        help=(
            "Agent type: mock (rule-based), hf (local HuggingFace model), "
            "openai (any OpenAI-compatible API — vLLM/Ollama/Together/etc.), "
            "claude (Anthropic API)."
        ),
    )
    parser.add_argument(
        "--model", default=None,
        help=(
            "Model ID or name. "
            "hf: HuggingFace ID or local checkpoint path. "
            "openai: model name as the server knows it (e.g. Qwen/Qwen2.5-7B-Instruct). "
            "claude: Claude model ID (default: claude-haiku-4-5-20251001)."
        ),
    )
    parser.add_argument(
        "--base-url", default=None, metavar="URL",
        help=(
            "Base URL for OpenAI-compatible inference server. "
            "Examples: http://localhost:8000 (vLLM), "
            "http://localhost:11434/v1 (Ollama), "
            "https://api.together.xyz/v1 (Together AI). "
            "Required for --agent openai."
        ),
    )
    parser.add_argument(
        "--api-key", default=None,
        help=(
            "API key / bearer token. "
            "For --agent claude: overrides ANTHROPIC_API_KEY env var. "
            "For --agent openai: use 'EMPTY' for local unauthenticated servers (default)."
        ),
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7).",
    )

    # Environment
    parser.add_argument(
        "--factory", default="auto_components_pune",
        choices=["auto_components_pune", "pharma_packaging_hyderabad", "textile_mill_surat"],
        help="Factory to run (default: auto_components_pune).",
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes to run (default: 3).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Base random seed. Each episode uses seed+episode_index (default: random).",
    )

    # Output
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-step details.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON lines (one per step) instead of pretty-print.",
    )

    # Remote
    parser.add_argument(
        "--remote", default=None, metavar="URL",
        help="Run against a remote server URL (e.g. http://localhost:7860).",
    )

    # Training
    train_grp = parser.add_argument_group("training (--agent hf only)")
    train_grp.add_argument(
        "--train", action="store_true",
        help="Interleave rollouts with GRPO updates (online training loop).",
    )
    train_grp.add_argument(
        "--train-only", action="store_true",
        help="Skip rollouts; train directly from an existing --buffer file.",
    )
    train_grp.add_argument(
        "--train-rounds", type=int, default=5,
        help="How many rollout→train rounds to run (default: 5).",
    )
    train_grp.add_argument(
        "--buffer", default="trajectories.jsonl", metavar="PATH",
        help="JSONL file to store/load trajectories (default: trajectories.jsonl).",
    )
    train_grp.add_argument(
        "--top-k", type=int, default=20,
        help="Use top-K episodes by reward for each GRPO update (default: 20).",
    )
    train_grp.add_argument(
        "--checkpoint-dir", default="./vishwakarma-grpo-output", metavar="DIR",
        help="Directory for GRPO checkpoints (default: ./vishwakarma-grpo-output).",
    )
    train_grp.add_argument(
        "--train-batch-size", type=int, default=4,
        help="Per-device batch size during GRPO training (default: 4).",
    )
    train_grp.add_argument(
        "--grad-accum", type=int, default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    train_grp.add_argument(
        "--learning-rate", type=float, default=1e-5,
        help="Learning rate for GRPO updates (default: 1e-5).",
    )
    train_grp.add_argument(
        "--train-epochs", type=int, default=1,
        help="Training epochs per GRPO round (default: 1).",
    )
    train_grp.add_argument(
        "--trainer", default="local",
        choices=["local", "vllm"],
        help=(
            "GRPO trainer backend. "
            "'local': in-process TRL (requires --agent hf). "
            "'vllm': send training jobs to a vLLM training server over HTTP "
            "(use with --agent openai or --agent hf; set --trainer-url)."
        ),
    )
    train_grp.add_argument(
        "--trainer-url", default=None, metavar="URL",
        help=(
            "Base URL of the vLLM training server (--trainer vllm only). "
            "Defaults to --base-url if --agent openai, else required. "
            "Example: http://localhost:8000"
        ),
    )
    train_grp.add_argument(
        "--trainer-poll", type=float, default=5.0, metavar="SECS",
        help="Seconds between vLLM training job status polls (default: 5).",
    )

    args = parser.parse_args()

    # ── build agent ──────────────────────────────────────────────────────────
    if args.agent == "mock":
        agent = MockAgent()
    elif args.agent == "hf":
        model_id = args.model or "Qwen/Qwen2.5-0.5B-Instruct"
        agent    = HFAgent(model_id=model_id, temperature=args.temperature)
    elif args.agent == "openai":
        if not args.base_url:
            sys.exit("--agent openai requires --base-url (e.g. http://localhost:8000)")
        model_id = args.model
        if not model_id:
            sys.exit("--agent openai requires --model (model name as the server knows it)")
        agent = OpenAICompatAgent(
            base_url    = args.base_url,
            model_id    = model_id,
            api_key     = args.api_key or "EMPTY",
            temperature = args.temperature,
        )
    elif args.agent == "claude":
        model_id = args.model or "claude-haiku-4-5-20251001"
        agent    = ClaudeAgent(model_id=model_id, api_key=args.api_key)
    else:
        sys.exit(f"Unknown agent: {args.agent}")

    do_train = (args.train or args.train_only)
    if do_train and args.trainer == "local" and args.agent != "hf":
        sys.exit("--trainer local requires --agent hf (model must be loaded in-process). "
                 "For remote training use --trainer vllm.")

    # ── build trainer ────────────────────────────────────────────────────────
    vllm_trainer = None
    if do_train and args.trainer == "vllm":
        trainer_url = args.trainer_url or args.base_url
        if not trainer_url:
            sys.exit("--trainer vllm requires either --trainer-url or --base-url")
        trainer_model = args.model or (agent.model_id if hasattr(agent, "model_id") else None)
        if not trainer_model:
            sys.exit("--trainer vllm requires --model (the model name on the training server)")
        vllm_trainer = VLLMGRPOTrainer(
            base_url      = trainer_url,
            model_id      = trainer_model,
            api_key       = args.api_key or "EMPTY",
            poll_interval = args.trainer_poll,
        )

    if not args.json:
        print(f"\nAgent      : {agent.name}")
        print(f"Factory    : {args.factory}")
        if not args.train_only:
            print(f"Episodes   : {args.episodes}")
        if do_train:
            print(f"Mode       : {'train-only' if args.train_only else 'online train'}")
            print(f"Trainer    : {args.trainer}"
                  + (f" @ {vllm_trainer.base_url}" if vllm_trainer else ""))
            print(f"Buffer     : {args.buffer}")
            print(f"Checkpoint : {args.checkpoint_dir}")

    # ── remote env mode (inference only) ─────────────────────────────────────
    if args.remote:
        run_remote(agent, args.remote, args.factory, args.episodes, args.verbose)
        return

    buffer    = TrajectoryBuffer(args.buffer) if do_train else None
    base_seed = args.seed if args.seed is not None else int(time.time()) % 10000

    def _do_grpo_update(rnd: int):
        """Dispatch one GRPO round to whichever trainer backend is configured."""
        if args.trainer == "vllm":
            vllm_trainer.train(
                buffer         = buffer,
                round_idx      = rnd,
                top_k          = args.top_k,
                batch_size     = args.train_batch_size,
                learning_rate  = args.learning_rate,
                epochs         = args.train_epochs,
                checkpoint_dir = args.checkpoint_dir,
            )
        else:
            run_grpo_training(
                hf_agent       = agent,
                buffer         = buffer,
                factory_id     = args.factory,
                checkpoint_dir = args.checkpoint_dir,
                round_idx      = rnd,
                top_k          = args.top_k,
                batch_size     = args.train_batch_size,
                grad_accum     = args.grad_accum,
                learning_rate  = args.learning_rate,
                epochs         = args.train_epochs,
            )

    # ── train-only: skip rollouts, go straight to GRPO ───────────────────────
    if args.train_only:
        existing = buffer.size() if buffer else 0
        if existing == 0:
            sys.exit(f"Buffer '{args.buffer}' is empty or missing. "
                     "Run some rollouts first (without --train-only).")
        print(f"\n[train-only] {existing} steps in buffer. Starting training.")
        _do_grpo_update(rnd=0)
        return

    # ── online training loop ─────────────────────────────────────────────────
    if args.train:
        all_results: list[EpisodeResult] = []
        ep_counter = 0

        for rnd in range(args.train_rounds):
            print(f"\n{'═'*72}")
            print(f"  ROUND {rnd+1}/{args.train_rounds}  —  rollout phase  ({args.episodes} episodes)")
            print(f"{'═'*72}")

            round_results = []
            for ep in range(args.episodes):
                seed = base_seed + ep_counter
                ep_counter += 1
                r = run_episode(
                    agent       = agent,
                    factory_id  = args.factory,
                    seed        = seed,
                    episode_idx = ep,
                    verbose     = args.verbose,
                    emit_json   = args.json,
                    buffer      = buffer,
                )
                round_results.append(r)
                all_results.append(r)

            if not args.json:
                print_aggregate(round_results)

            _do_grpo_update(rnd=rnd + 1)

        if not args.json and len(all_results) > args.episodes:
            print(f"\n{'═'*72}")
            print(f"  OVERALL ({len(all_results)} episodes across {args.train_rounds} rounds)")
            print_aggregate(all_results)
        return

    # ── plain inference (no training) ────────────────────────────────────────
    results: list[EpisodeResult] = []
    for ep in range(args.episodes):
        seed = base_seed + ep
        r    = run_episode(
            agent      = agent,
            factory_id = args.factory,
            seed       = seed,
            episode_idx= ep,
            verbose    = args.verbose,
            emit_json  = args.json,
            buffer     = buffer,
        )
        results.append(r)

    if not args.json:
        print_aggregate(results)


if __name__ == "__main__":
    # inference.py lives inside the repo root (which IS the vishwakarma_env package).
    # To import it as `vishwakarma_env`, we need the *parent* directory on sys.path.
    _repo_root   = os.path.dirname(os.path.abspath(__file__))
    _parent      = os.path.dirname(_repo_root)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    main()
