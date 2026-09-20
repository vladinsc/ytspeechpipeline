# Bengee four-A100 local JSON batch

This deployment runs eight independent workers: two batch-8 workers on each of
four 40 GB A100 GPUs. Every worker loads its own Demucs, Whisper `large-v3`, and
cached English alignment models. Silero VAD and Praat remain on CPU.

No database is required. Each completed video is written as one JSON file under
`results/kids` or `results/normal`. Eight separate checkpoint JSON files make
the run resumable without allowing workers to overwrite each other's state.

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
physical indexes. `BENGEE_WHISPER_BATCH_SIZE` defaults to 8.

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

Running the same command again resumes every shard. A completed entry is
reprocessed only if its result JSON is missing.

```bash
docker compose -f compose.bengee.yaml up -d
```

To retry failures in one shard:

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

Monitor GPU memory and utilization with:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv --loop=1
```

If a GPU approaches 38 GB, stop the deployment and lower
`BENGEE_WHISPER_BATCH_SIZE`. Do not run another GPU workload on the same cards
during the batch.
