# Implementation brief: MI300X pipeline, word-timed review videos, and YouTube upload

## Objective

Update this project so it can run reliably on one AMD Instinct MI300X (192 GB
VRAM, 20 vCPU, 240 GB RAM, 5 TB NVMe scratch) through ROCm, using Whisper
large-v3 in FP16. Add a review-video workflow that randomly selects **five
completed `kids` videos and five completed `normal` videos**, overlays each
transcript word over the source video at its exact aligned timing, and can
upload the resulting review videos to the owner's YouTube channel only when
explicitly requested.

Do not replace or break the existing NVIDIA/CUDA workflow. Make ROCm an
explicit, documented deployment option.

## Guardrails

- Treat external upload as opt-in. Rendering must be possible without any
  YouTube credentials. Upload must require an explicit `--upload` flag.
- Default every test upload to `private`; support `private`, `unlisted`, and
  `public` only through an explicit CLI option.
- Never log credentials, OAuth tokens, video URLs containing private data, or
  the contents of secret files.
- Do not put credentials, rendered MP4 files, downloaded media, work files, or
  OAuth token files under version control.
- Preserve the current resumable processing and checkpoint semantics.

## Part 1 — ROCm / MI300X architecture

### Target configuration

- Host: Linux with a ROCm release that supports MI300X.
- GPU: one MI300X. All GPU stages use that device.
- ASR: Whisper `large-v3`, `float16` (FP16). This is the requested
  non-INT8 configuration; do **not** use FP32 unless a benchmark proves a
  specific need.
- Initial throughput tuning values: one worker, then two workers; Whisper
  batch sizes 1, 4, 8, and 16. Select the production setting from measured
  throughput, stability, and transcript quality.

### Required changes

1. Add a ROCm-specific Dockerfile/Compose override or a clearly selected build
   path. Use a supported ROCm PyTorch image and expose the ROCm devices needed
   by the container (`/dev/kfd` and `/dev/dri`).
2. Use a HIP/ROCm-enabled CTranslate2 build compatible with the installed
   WhisperX version. Add a reproducible build/install path and pin tested
   versions.
3. Replace CUDA/NVIDIA-only checks in the ROCm path:
   - Do not invoke `nvidia-smi`.
   - Confirm PyTorch sees the HIP device, report its name and total/free
     memory, and fail with an actionable error if it does not.
   - Preserve CUDA checks in the existing NVIDIA deployment path.
4. Configure `WhisperX`, Demucs, and forced alignment to use the one logical
   ROCm GPU. PyTorch ROCm commonly retains the `cuda` device API; verify this
   with a smoke test rather than renaming device strings blindly.
5. Make compute type selectable. The ROCm FP16 batch command must pass
   `--compute-type float16`; the existing Compose batch command currently
   forces `int8` and must not silently override the requested value.
6. Update `.env.example`, Docker documentation, requirements, and smoke tests
   for the two deployment modes.

### Validation and benchmark gate

Run these in order and save a concise benchmark report under `results/`:

1. Smoke test model loading and one short video through every stage.
2. Run 10 representative videos; inspect transcript and word alignment.
3. Run 20–50 representative 5–10 minute videos for each useful combination:
   - 1 worker × batch 1/4/8/16
   - 2 workers × the best one-worker batch size
   - 4 workers only if two workers leave substantial GPU idle time
4. Capture videos/hour, real-time factor, CPU/RAM use, peak VRAM, GPU
   utilization, download failures, and processing failures.
5. Select the fastest stable configuration. Do not infer worker count from
   `192 GB / model size`: one GPU's compute is shared, so more workers are not
   automatically faster.

## Part 2 — Random review-video generator

### Input and selection

- Source candidates are successful JSON outputs in `results/kids` and
  `results/normal` (or configurable equivalent directories).
- Each JSON contains `source_url`, `label`, and a `words` array. Each word has
  `word`, `start_time`, and `end_time`; these are the authoritative timing
  fields.
- Add a command such as:

  ```text
  python review_videos.py generate --per-category 5 --seed 20260916
  ```

- Select exactly five unique successful videos from each category, for ten
  total **without asking the user to choose them**. Generate a seed by default;
  allow `--seed` only to reproduce a prior selection. Record selected source
  URLs, result JSON paths, and seed in a manifest JSON.
- Skip entries without usable word timing or whose source video cannot be
  downloaded. Replace skipped selections with another random eligible entry
  and record the reason.

### Media-retention policy

- Choose the ten review candidates deterministically from the labeled input
  manifests **before** their batch processing begins. Mark them in a persisted
  `review_selection.json` file so a resumed run retains the same random ten.
- For those ten selected videos only, retain the downloaded original audio
  track needed for review-video rendering in a dedicated, gitignored review
  source directory. Retain it until its corresponding MP4 has rendered and
  passed QA; allow a cleanup command to remove it afterward.
- For every non-selected video, delete downloaded audio, normalized WAV files,
  vocal stems, and other work files immediately after a successful JSON result
  is verified. Keep its JSON result and checkpoint entry.
- On failures, retain the work directory only when a `--keep-failed-workdirs`
  diagnostic option is supplied; otherwise clean it up after recording the
  error in the checkpoint.
- Never retain full source video files for the entire corpus. Download the
  original video for rendering only for the ten selected candidates, then
  remove it after its rendered MP4 passes QA unless an explicit keep option is
  given.

### Rendering contract

- Download the original source video at a stable, documented resolution.
- Render an MP4 copy with the original audio preserved.
- Overlay **one transcript word at a time**. For every item in `words`:
  - show its literal `word` text at exactly `start_time`;
  - remove it at exactly `end_time`;
  - do not extend it into the following pause or replace timings with a
    uniform/estimated cadence;
  - preserve punctuation/casing from the transcript JSON;
  - handle zero/invalid/overlapping timestamps deterministically and log the
    affected item.
- Use a readable, accessible subtitle style: high contrast text, opaque or
  semi-opaque background, safe lower-third placement, and resolution-aware
  sizing. Avoid covering the full frame.
- Use a frame-accurate approach. Generate ASS/SSA subtitle events or an
  equivalent FFmpeg-supported timed-subtitle format with millisecond timing,
  then burn it into the output with FFmpeg. Do not use timer-based GUI
  automation.
- Generate a sidecar `.ass`/`.srt` artifact and a render manifest per output
  so timing can be audited.
- Output layout:

  ```text
  results/review_videos/
    manifest_<timestamp>.json
    kids/
      <source-stem>_word_timed.mp4
      <source-stem>_word_timed.ass
      <source-stem>_render.json
    normal/
      ...
  ```

- Add a `--dry-run` mode that selects and writes manifests but does not
  download, render, or upload.

### Visual QA

- Add an automated timing test for synthetic words with known start/end times.
- For each rendered video, extract verification frames at word start, midpoint,
  and just after word end for several sampled words.
- Create a concise QA report showing expected versus actual subtitle event
  timing. Manually inspect at least one `kids` and one `normal` output before
  enabling uploads.

## Part 3 — YouTube upload

## Part 3a — Delivering the generated videos

YouTube is optional. The primary deliverable is the rendered MP4 file on the
machine that runs the pipeline.

- Keep all rendered files under `results/review_videos/` as specified above.
- Add a packaging command, for example:

  ```text
  python review_videos.py package results/review_videos/manifest_<timestamp>.json
  ```

  It must create `results/review_videos/review_set_<timestamp>.zip` containing
  the ten MP4 files, the selection/render manifest, and subtitle sidecars.
- Print the absolute paths, file sizes, and SHA-256 hashes of the MP4 files and
  ZIP archive at completion.
- Document safe retrieval for a remote MI300X machine, including an `scp` or
  `rsync` command that downloads only the generated ZIP archive to the user's
  computer. Do not expose arbitrary server directories over HTTP.
- Do not add Google Drive, S3, or another storage/account integration unless
  the user specifically chooses and authorizes that provider. YouTube upload is
  a separate optional publishing step described below.

### Authentication requirement

An API key is **not sufficient** to upload to a YouTube channel. Uploading
uses `videos.insert` and requires OAuth 2.0 authorization with the
`https://www.googleapis.com/auth/youtube.upload` scope. The owner must provide
an OAuth client secret (or an approved equivalent OAuth configuration) and
complete the one-time browser consent flow. Store the resulting refresh token
in a gitignored secret path.

Support environment variables like:

```dotenv
# API key may be useful for read-only API calls, but cannot authorize uploads.
YOUTUBE_API_KEY=

# Required for upload authorization.
YOUTUBE_OAUTH_CLIENT_SECRETS=/run/secrets/youtube_client_secret.json
YOUTUBE_OAUTH_TOKEN_PATH=/data/secrets/youtube_oauth_token.json
YOUTUBE_UPLOAD_PRIVACY=private
YOUTUBE_UPLOAD_NOTIFY_SUBSCRIBERS=false
```

Document the one-time OAuth setup and add these paths to `.gitignore`. Do not
ask the user to paste OAuth secrets into the repository or terminal output.

### Upload behavior

- Add a separate explicit command, for example:

  ```text
  python review_videos.py upload results/review_videos/manifest_<timestamp>.json \
    --upload --privacy private --confirm-count 10
  ```

- Require `--upload` and an exact `--confirm-count` to prevent accidental bulk
  uploads.
- Upload only rendered files listed in the supplied immutable manifest.
- Default metadata must clearly identify each video as a generated
  word-timing review artifact. Make title/description templates configurable.
- Set `notifySubscribers=false` by default.
- Persist upload status, YouTube video ID, returned URL, timestamp, privacy
  value, and errors in the manifest. The command must be idempotent: never
  upload the same completed artifact twice when rerun.
- Use resumable uploads, exponential-backoff retries for transient failures,
  and no retry for authorization/configuration errors.
- Keep initial uploads `private`. Do not publish public/unlisted videos unless
  the user explicitly selects that privacy level.

Note: uploads made from unverified YouTube API projects can be restricted to
private visibility. Surface this status clearly instead of treating it as a
pipeline failure.

## Completion criteria

The work is complete only when:

1. Existing CUDA mode remains functional.
2. ROCm MI300X mode completes the smoke test in FP16.
3. A benchmark report identifies the recommended worker/batch settings.
4. `generate --per-category 5` produces ten rendered videos, five per label,
   with word visibility matching the JSON `start_time`/`end_time` values.
5. Rendering and upload both resume safely after interruption.
6. Upload is impossible without explicit flags and OAuth credentials.
7. Documentation includes exact commands, required environment variables,
   output locations, and the Google OAuth setup steps.
