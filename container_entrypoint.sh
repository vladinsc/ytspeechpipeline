#!/usr/bin/env bash
set -euo pipefail

gpu_check() {
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
}

mode="${1:-api}"
case "$mode" in
  api)
    gpu_check
    shift || true
    exec -a yt-transcriber python -m uvicorn yt_transcriber_api:app \
      --host 0.0.0.0 --port "${YT_TRANSCRIBER_PORT:-8000}" --workers 1 "$@"
    ;;
  batch)
    gpu_check
    shift
    exec -a yt-transcriber python batch_pipeline.py "$@"
    ;;
  bengee-worker)
    gpu_check
    shift || true
    : "${BENGEE_WORKER_INDEX:?BENGEE_WORKER_INDEX is required}"
    worker_root="/data/work/worker-${BENGEE_WORKER_INDEX}"
    manifest_path="${worker_root}/videos_labeled.txt"
    mkdir -p "${worker_root}"
    {
      echo "# LABEL: KIDS"
      sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' /data/input/videos_kids.txt
      echo "# LABEL: NORMAL"
      sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' /data/input/videos_normal.txt
    } > "${manifest_path}.tmp"
    mv "${manifest_path}.tmp" "${manifest_path}"
    exec -a yt-transcriber python batch_pipeline.py "${manifest_path}" \
      --output-dir /data/results \
      --work-root "${worker_root}" \
      --shard-count "${BENGEE_WORKER_COUNT:-8}" \
      --shard-index "${BENGEE_WORKER_INDEX}" \
      --state-backend json \
      --device cuda \
      --demucs-device cuda \
      --alignment-device cuda \
      --whisper-model "${BENGEE_WHISPER_MODEL:-large-v3}" \
      --compute-type "${BENGEE_COMPUTE_TYPE:-int8}" \
      --batch-size "${BENGEE_WHISPER_BATCH_SIZE:-8}" \
      --language "${BENGEE_LANGUAGE:-en}" \
      --granularity "${BENGEE_GRANULARITY:-both}" \
      "$@"
    ;;
  shell)
    shift
    exec /bin/bash "$@"
    ;;
  *)
    echo "Usage: container_entrypoint.sh {api|batch|bengee-worker|shell}" >&2
    exit 2
    ;;
esac
