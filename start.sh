#!/bin/bash
# start.sh — HuggingFace Spaces entrypoint
#
# 1. Launches the Vishwakarma FastAPI env server on :7860 (background)
# 2. If API_BASE_URL + MODEL_NAME secrets are set, runs a 3-episode inference
#    demo using the configured OpenAI-compatible endpoint, writing results to
#    /tmp/inference_results.jsonl so they appear in Space logs.
#
# Required Space secrets (Settings → Variables and secrets):
#   API_BASE_URL   e.g. https://api-inference.huggingface.co/v1
#   MODEL_NAME     e.g. Qwen/Qwen2.5-72B-Instruct
#   HF_TOKEN       your HuggingFace write token (used as API key)

set -euo pipefail

echo "=== Vishwakarma Factory Environment ==="
echo "Factory  : ${FACTORY_ID:-auto_components_pune}"
echo "API URL  : ${API_BASE_URL:-(not set — inference demo will be skipped)}"
echo "Model    : ${MODEL_NAME:-(not set)}"
echo "========================================"

# ── 1. Start the FastAPI env server ─────────────────────────────────────────
echo "[server] Starting env server on port 7860 …"
uvicorn vishwakarma_env.server.app:app \
    --host 0.0.0.0 \
    --port 7860 \
    --log-level info &
SERVER_PID=$!

# Wait for the server to be ready (poll /health up to 30s)
echo "[server] Waiting for health check …"
for i in $(seq 1 30); do
    if curl -sf http://localhost:7860/health > /dev/null 2>&1; then
        echo "[server] Ready."
        break
    fi
    sleep 1
done

# ── 2. Run inference demo if secrets are configured ─────────────────────────
if [ -n "${API_BASE_URL:-}" ] && [ -n "${MODEL_NAME:-}" ]; then
    echo ""
    echo "[inference] Running 3-episode demo …"
    echo "[inference] Endpoint : ${API_BASE_URL}"
    echo "[inference] Model    : ${MODEL_NAME}"
    echo "[inference] Factory  : ${FACTORY_ID:-auto_components_pune}"
    echo ""

    python inference.py \
        --agent openai \
        --base-url  "${API_BASE_URL}" \
        --model     "${MODEL_NAME}" \
        --api-key   "${HF_TOKEN:-EMPTY}" \
        --factory   "${FACTORY_ID:-auto_components_pune}" \
        --episodes  3 \
        --seed      42 \
        --verbose \
        --buffer    /tmp/trajectories.jsonl \
        2>&1 | tee /tmp/inference_results.txt

    echo ""
    echo "[inference] Results written to /tmp/inference_results.txt"
    echo "[inference] Trajectories written to /tmp/trajectories.jsonl"
else
    echo "[inference] API_BASE_URL / MODEL_NAME not set — skipping inference demo."
    echo "[inference] Set them in Space Settings → Variables and secrets to enable."
fi

# ── 3. Keep the server alive ─────────────────────────────────────────────────
echo ""
echo "[server] Env server running (PID ${SERVER_PID}). Waiting …"
wait $SERVER_PID
