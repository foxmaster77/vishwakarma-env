FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir fastapi uvicorn httpx pydantic

# Copy source
COPY vishwakarma_env/ ./vishwakarma_env/

# Expose port
EXPOSE 7860

# Environment variables
ENV FACTORY_ID=auto_components_pune
ENV PYTHONPATH=/app

# Run server
CMD ["uvicorn", "vishwakarma_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
