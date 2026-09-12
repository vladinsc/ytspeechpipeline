# GPU container

The pipeline is packaged as one worker spanning two GPUs. WhisperX runs with
INT8 quantization and batch size 1 on the primary GPU. Demucs and forced
alignment run on the secondary GPU. Silero VAD and Praat run on CPU. Model
weights are baked into the image and reused for every video.

The image also includes Deno and the matching `yt-dlp-ejs` package. They let
yt-dlp solve the JavaScript challenges currently required by many public
YouTube streams; no YouTube API token or browser cookie is required for normal
public videos.

## GPU server prerequisites

- Linux x86-64
- An NVIDIA GPU whose driver supports CUDA 12.8 containers
- Docker Engine
- NVIDIA Container Toolkit configured for Docker

Verify GPU passthrough before building:

```bash
nvidia-smi
docker run --rm --gpus all pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Build

The first build downloads all public model weights and produces a large image:

```bash
docker compose build
```

To build a smaller image that downloads models on first execution instead:

```bash
docker build --build-arg PRELOAD_MODELS=0 -t yt-transcriber:cuda12.8 .
```

## Configure the API and Traefik

Create the local environment file:

```bash
cp .env.example .env
```

Select two physical GPUs shown by `nvidia-smi` and the Docker network belonging
to the Traefik instance that serves `ai.asigno.ro`:

```dotenv
YT_TRANSCRIBER_GPU_ID=6
YT_TRANSCRIBER_SECONDARY_GPU_ID=7
TRAEFIK_NETWORK=traefik_traefik_net
```

Compose exposes the two physical GPUs in that order. Inside the container, the
primary GPU is renumbered as CUDA device `0` and the secondary GPU as `1`.
Compose assigns WhisperX to `cuda:0` (written as `cuda`), and Demucs plus forced
alignment to `cuda:1`.

Ensure the external Docker network named by `TRAEFIK_NETWORK` already contains
Traefik. Merge `traefik-dynamic-yt-transcriber.yml` into the corresponding
`http.routers`, `http.services`, and `http.middlewares` sections of your existing
Traefik file-provider configuration.

Start the long-running API:

```bash
mkdir -p results _batch_work
docker compose up -d yt-transcriber-api
```

The Compose project is `yt-transcriber`, the routed service/container is
`yt-transcriber-api`, and Traefik forwards to port 8000. The main process uses
the command name `yt-transcriber`.

## Submit an asynchronous job

```bash
curl -X POST "https://ai.asigno.ro/yt-transcriber/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID","label":"kids","granularity":"both"}'
```

The API immediately returns HTTP 202 and a job ID. Check it without holding the
original request open:

```bash
curl "https://ai.asigno.ro/yt-transcriber/v1/jobs/JOB_ID"

curl "https://ai.asigno.ro/yt-transcriber/v1/jobs/JOB_ID/result"
```

API job state is persisted in `results/api_jobs.json`; labeled result JSON files
are stored under `results/kids` and `results/normal`. Successful job audio is
removed automatically.

## Run the manifest batch

The original labeled `videos.txt` workflow remains available as a separate
profile. Do not run it alongside the API because both services use the same two
GPUs:

```bash
docker compose --profile batch run --rm yt-transcriber-batch
```

The batch worker reuses the existing `TRAEFIK_NETWORK`; Compose therefore does
not create a per-project default bridge network on the shared GPU host.

Resume with the same command. To retry failed manifest entries:

```bash
docker compose --profile batch run --rm yt-transcriber-batch \
  batch /data/input/videos.txt \
  --output-dir /data/results \
  --work-root /data/work \
  --device cuda --demucs-device cuda:1 --alignment-device cuda:1 \
  --compute-type int8 --batch-size 1 \
  --language en --granularity both --retry-failed
```

At startup, the container runs `nvidia-smi` and a PyTorch CUDA check. It exits
before loading any models if GPU passthrough is unavailable. The initial GPU
snapshot is also stored in the relevant job checkpoint. For live GPU
monitoring on the host, use:

```bash
watch -n 2 nvidia-smi
```

Inspect progress from the host:

```bash
python -c 'import json; print(json.load(open("results/batch_checkpoint.json"))["summary"])'
```

The endpoints are intentionally open and do not require application credentials.
The container does not include YouTube credentials or Hugging Face tokens. The
selected videos and model repositories are public.
