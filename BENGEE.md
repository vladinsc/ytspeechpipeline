# Bengee four-GPU batch

This deployment runs four independent copies of the existing pipeline in
parallel. Each container sees one NVIDIA GPU and runs Demucs, quantized Whisper
`large-v3`, and forced alignment on that GPU. Silero VAD and Praat remain on the
CPU. The manifest is assigned round-robin, so every source video belongs to
exactly one worker.

The services share the final `results/kids` and `results/normal` directories but
use separate checkpoints and work directories:

| Service | GPU setting | Manifest indexes | Checkpoint |
| --- | --- | --- | --- |
| `yt-transcriber-batch-0` | `BENGEE_GPU_0` | 1, 5, 9, ... | `batch_checkpoint_bengee_0.json` |
| `yt-transcriber-batch-1` | `BENGEE_GPU_1` | 2, 6, 10, ... | `batch_checkpoint_bengee_1.json` |
| `yt-transcriber-batch-2` | `BENGEE_GPU_2` | 3, 7, 11, ... | `batch_checkpoint_bengee_2.json` |
| `yt-transcriber-batch-3` | `BENGEE_GPU_3` | 4, 8, 12, ... | `batch_checkpoint_bengee_3.json` |

## Configure and run

Verify that Docker can see all four GPUs:

```bash
nvidia-smi
docker run --rm --gpus all pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime \
  python -c 'import torch; print(torch.cuda.device_count())'
```

Copy `.env.example` to `.env`. If the physical GPU indexes are not `0` through
`3`, change `BENGEE_GPU_0` through `BENGEE_GPU_3` in `.env`.

```bash
mkdir -p results _batch_work
docker compose -f compose.bengee.yaml build
docker compose -f compose.bengee.yaml up -d
docker compose -f compose.bengee.yaml logs -f
```

Stopping and running the same `up -d` command resumes pending work from the four
checkpoint files. Completed entries are skipped while their result JSON exists.

Inspect every worker's status:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("results").glob("batch_checkpoint_bengee_*.json")):
    data = json.loads(path.read_text())
    print(path.name, data.get("summary", {}))
PY
```

Do not run the original two-GPU API or batch service at the same time. It may
claim the same GPUs or write the same result names.

Every worker explicitly uses:

- Whisper model `large-v3`
- CTranslate2 compute type `int8`
- Whisper batch size `1`
- its one visible `cuda` device for WhisperX, Demucs, and alignment
- one of four deterministic manifest shards
