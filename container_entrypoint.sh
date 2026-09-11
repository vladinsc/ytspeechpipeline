#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${YT_TRANSCRIBER_API_KEY:-}" ]]; then
  key_file="${YT_TRANSCRIBER_API_KEY_FILE:-/data/results/.yt-transcriber-api-key}"
  mkdir -p "$(dirname "$key_file")"
  umask 077
  if [[ ! -s "$key_file" ]]; then
    temporary_key="${key_file}.tmp"
    python -c "import secrets; print(secrets.token_hex(32))" > "$temporary_key"
    mv "$temporary_key" "$key_file"
    echo "[api-key] Generated a persistent API key at $key_file"
  else
    echo "[api-key] Reusing the persistent API key at $key_file"
  fi
  YT_TRANSCRIBER_API_KEY="$(tr -d '\r\n' < "$key_file")"
  export YT_TRANSCRIBER_API_KEY
fi

echo "[gpu-check] NVIDIA devices visible inside the container:"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu-check] ERROR: nvidia-smi is unavailable. Install/configure NVIDIA Container Toolkit on the host." >&2
  exit 2
fi
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu \
  --format=csv

python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("[gpu-check] ERROR: PyTorch cannot access CUDA inside the container.")
print(f"[gpu-check] PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
print(f"[gpu-check] Selected GPU: {torch.cuda.get_device_name(0)}")
PY

mode="${1:-api}"
case "$mode" in
  api)
    shift || true
    exec -a yt-transcriber python -m uvicorn yt_transcriber_api:app \
      --host 0.0.0.0 --port "${YT_TRANSCRIBER_PORT:-8000}" --workers 1 "$@"
    ;;
  batch)
    shift
    exec -a yt-transcriber python batch_pipeline.py "$@"
    ;;
  shell)
    shift
    exec /bin/bash "$@"
    ;;
  *)
    echo "Usage: container_entrypoint.sh {api|batch|shell}" >&2
    exit 2
    ;;
esac
