# Newman/Ratner transcript and feature path

Newman/Ratner has CHAT (`.cha`) transcripts. The primary route uses trusted CHAT
text with WhisperX CTC word alignment in a small audio window around each CHAT
time link. It extracts the same Parselmouth prosody set as the main pipeline:
utterance F0 and intensity summaries, word and estimated-syllable F0, pauses,
speech rate, word rate, and articulation coverage. It writes no phones because
WhisperX provides word boundaries only.

The same run can create a separate full-recording, Silero-VAD WhisperX ASR
artifact. That route produces its own segment times and text, then compares them
with CHAT as a diagnostic. It never replaces CHAT text, timings, or tiers.

## Convert one recording now

From the repository root:

```powershell
python newman_ratner_pipeline.py --age 07 --recording 4269LP
```

This needs only Python's standard library. It writes
`results/newman_ratner/07/4269LP/play_transcript.json` and
`interview_transcript.json`. Each JSON contains the CHAT headers, all speaker
turns, their original CHAT text and dependent tiers (including `%mor` and
`%gra`), time links, and a filtered `utterances` list for the selected speaker.
The default speaker is `MOT` (mother). The selected list excludes turns with
missing or multiple time links, unintelligible text, durations over 25 seconds,
and implausible text rates over 8 words/second. Exclusion counts are in
`conversion_counts`. The full `all_turns` list preserves excluded turns for
review. The conversion does not calculate syntax features from `%mor`/`%gra`.

## Align CHAT and extract features on Bengee

Use the Bengee GPU environment with WhisperX, NumPy, Parselmouth, FFmpeg, and
ffprobe. The dedicated Compose job uses one GPU and runs both tracks for a
reviewable 4269LP play pilot. After copying this repository and the Newman/Ratner
media/CHAT tree to Bengee with the same relative paths, run:

```bash
docker compose -f compose.bengee.newman.yaml config --quiet
docker compose -f compose.bengee.newman.yaml build
docker compose -f compose.bengee.newman.yaml run --rm newman-pilot \
  features --sources NewmanRatner \
  --source-id NewmanRatner:07:4269LP:play \
  --newman-root data/staging/newman_ratner --dry-run
docker compose -f compose.bengee.newman.yaml up --abort-on-container-exit
```

The bind-mounted data must include
`data/staging/newman_ratner/media/07/4269LP.wav` and
`data/staging/newman_ratner/transcripts/extracted/07/4269LP.cha`.
`BENGEE_NEWMAN_GPU` selects the physical GPU (default `0`); inside the
container it appears as `cuda:0`. Results persist under
`results/newman_ratner_gpu_pilot/`. The Docker build preloads the English
WhisperX alignment model alongside the ASR and Silero models; the image build
requires model-download access.

For a direct Python environment, the equivalent command is:

```powershell
python newman_ratner_pipeline.py `
  --age 07 --recording 4269LP `
  --align --alignment-backend whisperx_chat `
  --validate-independent-asr `
  --device cuda:0 --alignment-device cuda:0 `
  --batch-size 8 --max-utterances 20 `
  --out-dir results/newman_ratner_whisperx_pilot
```

Remove `--max-utterances 20` after visually and aurally reviewing the pilot.
Each alignment clip includes 0.25 seconds of context by default; change it with
`--chat-context-sec` if review shows clipped edge words. Prosody summaries still
cover only the original CHAT interval. The full independent ASR pass uses
WhisperX with Silero VAD and stores VAD-derived segments and CTC word times in
`*_independent_asr_validation.json`.

Output is one JSON object per mother utterance in
`play_whisperx_chat_features.jsonl` or `interview_whisperx_chat_features.jsonl`.
Every record retains CHAT text, dependent tiers, CHAT link, actual alignment
search window, and source-timeline word times. Results append per batch and a
companion `.config.json` prevents mixing changed inputs or settings. The status
file also points to the independent validation artifact when requested.

`target_speaker_overlap_summary.wer` in that artifact is a screening measure,
not a corpus-wide accuracy claim: an independent ASR segment can include another
speaker, and CHAT conventions can differ from ASR spelling. Review its worst
turns with the alignment video and audio.

## Legacy MFA comparison route

MFA remains available only as an explicit comparison route. Install MFA and its
English models as described in [MFA_ALIGNMENT.md](MFA_ALIGNMENT.md), then use
`--alignment-backend mfa --mfa-conda-env speech-features`. Keep its result in a
separate output directory. The visually rejected 4269LP MFA pilot must not be
used for thesis timing analyses.

## Three-recording quality pilot on Bengee

`compose.bengee.newman.quality.yaml` runs a small, reviewable pilot with three
recordings: `07/4269LP` play, `18/5694MC` interview, and `24/7120CB` interview.
The audio totals about 43 minutes. CHAT-guided WhisperX alignment and the full
prosody feature set run on a spread of at most 12 turns per available speaker
tier in each recording. Silero-VAD WhisperX ASR and pyannote diarization run on
the full audio. The independent ASR and diarization receive no CHAT words,
turn times, or speaker labels. CHAT is used afterward as an evaluation
reference. This pilot uses one GPU; the broader transcription jobs can use the
other GPUs separately.

The pyannote `speaker-diarization-community-1` model requires a Hugging Face
read token whose account has accepted that model's access terms. On Bengee,
put the token in this project's `.env` file only:

```dotenv
HF_TOKEN=your_hugging_face_read_token
# Optional: select the physical GPU assigned to this pilot
BENGEE_NEWMAN_GPU=0
```

Do not commit `.env`. Git and Docker ignore it. Docker Compose reads it from
the project directory and passes `HF_TOKEN` only to the pilot container; this
does not set a system-wide variable on Bengee. Restrict the file to your
account (`chmod 600 .env`). The token can still be seen by people with Docker
administrative access on the host, so use a read-only token.

The data tree must exist on Bengee at `data/staging/newman_ratner`, including
both `media` and `transcripts/extracted` trees. Git transfer moves code, not
the ignored audio tree. From the project root on Bengee:

```bash
docker compose -f compose.bengee.newman.quality.yaml config --quiet
docker compose -f compose.bengee.newman.quality.yaml build quality-pilot
docker compose -f compose.bengee.newman.quality.yaml run --rm quality-pilot --dry-run
docker compose -f compose.bengee.newman.quality.yaml up --abort-on-container-exit
```

Use `config --quiet` because the ordinary config output would expose the
expanded token. Results appear under `results/newman_quality_pilot`, with one
directory per source. Each contains `report.json`, `diarization.json`,
`independent_asr_validation.json`, `speaker_attributed_asr.json`,
`chat_guided_features.jsonl`, and one or two numbered review videos covering
the sampled turn times. The video shows the supplied CHAT words, independently
recognized text, CHAT speaker tier, audio-only speaker cluster, alignment, and
prosody. Cluster-to-CHAT mapping in the report is computed only for scoring.

`speaker_coverage` and mapped accuracy measure diarization on exclusive timed
CHAT spans; they are not full-corpus diarization error rate. The ASR WER is a
diagnostic based on word midpoints in non-overlapping CHAT turns. CHAT
conventions, inaccurate turn boundaries, and overlapping voices can affect
both scores. `aligned_word_coverage` says how many words got timestamps; it
does not establish that those timestamps are correct. Listen to the videos and
inspect errors before scaling up.

## Corpus quality review before audience labels

The locally staged play and interview CHAT files often have nearly identical
speaker text despite different media time links. For `4269LP`, transcript text
overlap is 100%. The converter records `pair_chat_text_overlap` and marks such
pairs `near_duplicate_CHAT_text_do_not_compare`. The `condition` field reports
the media folder name only. Do not use it as a verified child-directed or
adult-directed label until the audio and source annotations are reviewed.
Alignment gives word/phone times for a supplied transcript; it does not prove
that the transcript matches the selected audio. Inspect a small set of clips
and alignments before a full corpus run.
