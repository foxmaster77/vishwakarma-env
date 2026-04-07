# HuggingFace Deployment Guide

Deploy vishwakarma-env to HuggingFace Spaces in 4 steps.

---

## Step 1 — Install OpenEnv CLI

```bash
pip install openenv-core
```

---

## Step 2 — Login to HuggingFace

```bash
huggingface-cli login
# Enter your HuggingFace token from https://huggingface.co/settings/tokens
```

---

## Step 3 — Set your API key (optional but recommended)

```bash
# Adds Claude reasoning scorer — improves Round 1 LLM evaluation score
export ANTHROPIC_API_KEY=sk-ant-...
```

To set it permanently in your HF Space:
1. Go to your Space settings on HuggingFace
2. Click "Variables and secrets"
3. Add `ANTHROPIC_API_KEY` as a secret

---

## Step 4 — Deploy

```bash
cd vishwakarma_env
openenv push --repo-id YOUR-HF-USERNAME/vishwakarma-env
```

That's it. Your environment will be live at:
`https://tiny2520tots-vishwakarma-env.hf.space`

---

## Verify it's working

```bash
curl https://tiny2520tots-vishwakarma-env.hf.space/health
# Should return: {"status": "ok", "factory": "auto_components_pune"}
```

```bash
curl -X POST https://tiny2520tots-vishwakarma-env.hf.space/reset
# Should return full observation JSON
```

---

## Test with the remote demo

```bash
python examples/grpo_training.py --base-url https://tiny2520tots-vishwakarma-env.hf.space
```

---

## Different factory configurations

The Dockerfile supports a `FACTORY_ID` environment variable.
Deploy three separate Spaces for three factories:

```bash
# Pune auto components (default)
openenv push --repo-id tiny2520tots/vishwakarma-env-pune

# Hyderabad pharma (set FACTORY_ID in Space settings)
openenv push --repo-id tiny2520tots/vishwakarma-env-pharma
# Then in Space settings: FACTORY_ID = pharma_packaging_hyderabad

# Surat textile
openenv push --repo-id tiny2520tots/vishwakarma-env-textile
# Then in Space settings: FACTORY_ID = textile_mill_surat
```

---

## Submission checklist (Round 1, due April 8)

- [ ] Environment deployed to HuggingFace Spaces
- [ ] `/health` endpoint returns 200
- [ ] `/reset` returns valid observation JSON
- [ ] `/step` with a simple action returns reward
- [ ] `openenv.yaml` in repo root
- [ ] README.md describes the environment
- [ ] At least 10 test episodes run without error
- [ ] `ANTHROPIC_API_KEY` secret set in Space (for reasoning scorer)

---

## Troubleshooting

**Space fails to build:**
Check the Docker build logs. Most common issue is missing `data/` directory.
Make sure all `.json` files in `data/` are included in the zip.

**Timeout on first request:**
HuggingFace Spaces have a cold start delay (~30 seconds). Wait and retry.

**`openenv push` not found:**
Run `pip install openenv-core` first. Then `openenv init` won't work
on an existing directory — use `openenv push` directly from the env directory.
