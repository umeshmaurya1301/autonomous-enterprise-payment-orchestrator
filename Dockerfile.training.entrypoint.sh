#!/usr/bin/env bash
# Entrypoint for the AEPO GRPO Training Space.
# 1. Run training (this is the expensive step — A10G, ~35 min)
# 2. Serve results on port 7860 so you can inspect/download them.
set -euo pipefail

echo "=== AEPO GRPO Training Space starting ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"

# Run training — pushes LoRA adapter to HF Hub if HF_TOKEN is set
python train_grpo_hf.py

echo "=== Training complete. Serving results on port 7860 ==="

# Minimal static file server so you can view/download results from the Space UI
python -c "
import http.server, os, pathlib
os.chdir('/app')
handler = http.server.SimpleHTTPRequestHandler
with http.server.HTTPServer(('0.0.0.0', 7860), handler) as httpd:
    print('Serving /app on port 7860 — open /results/grpo_reward_curve.png to verify training.')
    httpd.serve_forever()
"
