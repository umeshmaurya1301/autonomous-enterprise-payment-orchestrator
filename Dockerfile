# ─────────────────────────────────────────────────────────────────────────────
# Unified Fintech Risk Gateway (UFRG) — Production Container
# ─────────────────────────────────────────────────────────────────────────────
# Two usage modes:
#
#   1. API server (default — used by HF Spaces and openenv validate):
#        docker run -p 7860:7860 ufrg
#
#   2. Inference / baseline scoring (used by evaluators for baseline scores):
#        docker run --rm \
#          -e SPACE_URL=http://localhost:7860 \
#          -e DRY_RUN=true \
#          ufrg python inference.py
#
#      To run against the live HF Space:
#        docker run --rm \
#          -e SPACE_URL=https://huggingface.co/spaces/unknown1321/unified-fintech-risk-gateway \
#          -e HF_TOKEN=hf_... \
#          ufrg python inference.py
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="Umesh Maurya <umeshmaurya1301>" \
    description="Unified Fintech Risk Gateway — Gymnasium RL Environment" \
    version="1.0.0"

# ── OS-level hardening ───────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Copy ALL application source (including server folder) ────────────────────
COPY . /app

# ── Install dependencies from requirements.txt ───────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Port configuration ───────────────────────────────────────────────────────
# Expose the exact port Hugging Face Spaces routes traffic to
EXPOSE 7860

# ── Default entrypoint: Start the FastAPI server ─────────────────────────────
# This is the default used by Hugging Face Spaces and openenv validate.
#
# To run the baseline inference script instead (reproduces dry-run scores):
#   docker run --rm -e DRY_RUN=true ufrg python inference.py
#
# To run against the live HF Space with a real LLM:
#   docker run --rm \
#     -e SPACE_URL=https://<your-space>.hf.space \
#     -e HF_TOKEN=hf_... \
#     -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
#     ufrg python inference.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]