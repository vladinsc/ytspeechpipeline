# Bengee four-A100 local JSON batch

This deployment runs eight independent workers: two batch-4 workers on each of
four 40 GB A100 GPUs. Every worker loads its own Demucs, Whisper `large-v3`, and
cached English alignment models. Silero VAD and Praat remain on CPU.

No database is required. Each completed video is written as one JSON file under
`results/kids` or `results/normal`. Eight separate checkpoint JSON files make
the run resumable without allowing workers to overwrite each other's state.
The workers reuse Docker's built-in bridge network because Bengee's predefined
address pools are exhausted; Compose therefore does not allocate a new subnet.

## Input selection

Compose reads:

- `artifacts/videos_selected_kids_1000.txt`
- `artifacts/videos_selected_youtube_1000.txt`

At startup, every worker generates the same labeled manifest and selects its
own deterministic round-robin shard. The two files contain 2,000 labeled items.
There are 35 URLs present in both classes; these are intentionally processed
once as `kids` and once as `normal`, producing distinct result files.

## Run

```bash
mkdir -p results _batch_work
docker compose -f compose.bengee.yaml build
docker compose -f compose.bengee.yaml up -d
docker compose -f compose.bengee.yaml ps -a
docker compose -f compose.bengee.yaml logs -f
```

The default GPU mapping is:

| GPU | Workers |
|---:|---:|
| 0 | 0, 1 |
| 1 | 2, 3 |
| 2 | 4, 5 |
| 3 | 6, 7 |

Override `BENGEE_GPU_0` through `BENGEE_GPU_3` if `nvidia-smi` reports different
physical indexes. `BENGEE_WHISPER_BATCH_SIZE` defaults to 4. Real measurements
with two batch-8 workers reached 36-37 GB on 40 GB A100s and left too little
headroom for transient allocations.

## YouTube cookies

If YouTube responds with "Sign in to confirm you're not a bot", export only
`youtube.com` cookies in Netscape format and treat the file like a password.
The yt-dlp guidance recommends exporting from a private/incognito YouTube
session and warns that automated use can put the account at risk; a dedicated
account is safer.

Store the exported file outside Git and restrict its permissions:

```bash
mkdir -p .secrets
install -m 600 /path/to/youtube-cookies.txt .secrets/youtube-cookies.txt
```

Enable it in the ignored `.env` file:

```dotenv
YT_DLP_COOKIES_FILE=/run/secrets/youtube-cookies.txt
```

Compose mounts `.secrets` read-only. The directory is excluded from both Git
and the Docker build context, so cookies are not committed or copied into an
image. Test the cookie before restarting the batch:

```bash
docker compose -f compose.bengee.yaml run --rm \
  yt-transcriber-batch-0 shell -lc \
  'yt-dlp --cookies /run/secrets/youtube-cookies.txt --simulate \
  "https://www.youtube.com/watch?v=2vYqjQnm3WY"'
```

## Output and checkpoints

Successful output files use globally unique manifest indexes:

```text
results/kids/video_0001_VIDEO_ID.json
results/normal/video_1001_VIDEO_ID.json
```

Checkpoint files are:

```text
results/batch_checkpoint_shard_0_of_8.json
...
results/batch_checkpoint_shard_7_of_8.json
```

Intermediate media is stored under `_batch_work/worker-N` and removed after a
successful video unless `--keep-workdirs` is supplied.

## Resume and retry

Running the same command again resumes every shard. The Bengee services pass
`--retry-failed`, so each failed entry is attempted once per deployment. It
does not loop repeatedly inside a worker. A completed entry is reprocessed only
if its result JSON is missing.

```bash
docker compose -f compose.bengee.yaml up -d
```

To retry only one shard manually:

```bash
docker compose -f compose.bengee.yaml run --rm \
  yt-transcriber-batch-0 bengee-worker --retry-failed
```

Repeat with the corresponding service for other failed shards.

## Inspect aggregate progress

```bash
jq -s '
  {
    total:      (map(.summary.total)      | add),
    pending:    (map(.summary.pending)    | add),
    processing: (map(.summary.processing) | add),
    completed:  (map(.summary.completed)  | add),
    failed:     (map(.summary.failed)     | add)
  }
' results/batch_checkpoint_shard_*_of_8.json
```

## Long videos and GPU memory

The complete recording and Demucs' multi-stem output accumulator remain in
system RAM. Demucs still runs each split on CUDA. This prevents multi-hour
recordings from allocating a full-track tensor on the GPU while preserving GPU
acceleration. Bengee has ample RAM for two simultaneous accumulators per GPU.

After pulling a version containing this fix, rebuild the image. The mounted
results and checkpoint files are preserved:

```bash
docker compose -f compose.bengee.yaml down
docker compose -f compose.bengee.yaml up -d --build
```

Monitor GPU memory and utilization with:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv --loop=1
```

If a GPU approaches 38 GB during transcription, lower
`BENGEE_WHISPER_BATCH_SIZE`. If it still approaches 38 GB during vocal
isolation after this fix, stop one worker on each affected GPU and finish those
four shards in a second wave. Do not run unrelated GPU workloads on the same
cards during the batch.
