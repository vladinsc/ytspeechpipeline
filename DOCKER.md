# GPU container

The pipeline is packaged as one GPU worker because Demucs, VAD, WhisperX,
alignment, and Praat run sequentially on the same audio. Model weights are baked
into the image and reused for every video.

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
curl -X POST "https://bengee.asigno.ro/yt-transcriber/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID","label":"kids","granularity":"both"}'
```

The API immediately returns HTTP 202 and a job ID. Check it without holding the
original request open:

```bash
curl "https://bengee.asigno.ro/yt-transcriber/v1/jobs/JOB_ID"

curl "https://bengee.asigno.ro/yt-transcriber/v1/jobs/JOB_ID/result"
```

API job state is persisted in `results/api_jobs.json`; labeled result JSON files
are stored under `results/kids` and `results/normal`. Successful job audio is
removed automatically.

## Run the manifest batch

The original labeled `videos.txt` workflow remains available as a separate
profile. Do not run it alongside the API on the same GPU:

```bash
docker compose --profile batch run --rm yt-transcriber-batch
```

Resume with the same command. To retry failed manifest entries:

```bash
docker compose --profile batch run --rm yt-transcriber-batch \
  batch /data/input/videos.txt \
  --output-dir /data/results \
  --work-root /data/work \
  --device cuda --language en --granularity both --retry-failed
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
