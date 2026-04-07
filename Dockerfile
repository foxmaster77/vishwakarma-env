FROM python:3.11-slim

WORKDIR /app

# ── system deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ── python deps ──────────────────────────────────────────────────────────────
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi uvicorn httpx pydantic \
    openai anthropic

# ── source ───────────────────────────────────────────────────────────────────
COPY vishwakarma_env/ ./vishwakarma_env/
COPY inference.py     ./inference.py
COPY start.sh         ./start.sh
RUN chmod +x start.sh

# ── runtime config ───────────────────────────────────────────────────────────
EXPOSE 7860
ENV FACTORY_ID=auto_components_pune
ENV PYTHONPATH=/app

# HF Space secrets — set these in Space Settings → Variables and secrets:
#   HF_TOKEN      your HuggingFace write token (for private model access)
#   API_BASE_URL  your inference endpoint base URL
#   MODEL_NAME    model name as the endpoint knows it
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

CMD ["./start.sh"]
