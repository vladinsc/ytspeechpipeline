"""Transcribe staged, untranscribed recordings without running forced alignment.

Three independent workers can process deterministic, duration-balanced shards.
Outputs keep source-timeline segment times for later review and CPU alignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
from pathlib import Path


LOG = logging.getLogger("transcription-batch")
ASR_STATUSES = {"whisperx_required", "asr_required", "no_transcript"}
DEFAULT_TIERS = {"acoustic_primary", "narration_supplement", "adult_supplement"}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def output_key(source_id: str) -> str:
    digest = hashlib.sha256(source_id.encode()).hexdigest()[:16]
    return digest


def inventory(manifest: Path, root: Path, sources: set[str], include_holds: bool) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    summary: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    with manifest.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            dataset = item.get("dataset", "unknown")
            info = summary.setdefault(dataset, {"eligible": 0, "has_transcript": 0,
                                                "not_staged": 0, "hold": 0,
                                                "unsupported_language": 0,
                                                "untranscribed_other": 0})
            if item.get("acquisition_status") != "staged":
                info["not_staged"] += 1
                continue
            status = item.get("transcript_status", "")
            if status not in ASR_STATUSES:
                info["untranscribed_other" if status == "no_lexical_transcript" else "has_transcript"] += 1
                continue
            if item.get("language") != "en":
                info["unsupported_language"] += 1
                continue
            held = item.get("usage_tier") not in DEFAULT_TIERS or item.get("rights_review") == "hold"
            if held and not include_holds:
                info["hold"] += 1
                continue
            audio = root / item["relative_path"]
            if not audio.is_file():
                info["not_staged"] += 1
                continue
            info["eligible"] += 1
            if dataset not in sources:
                continue
            source_id = item["source_id"]
            if source_id in seen:
                raise ValueError(f"Duplicate source_id: {source_id}")
            seen.add(source_id)
            rows.append({"source_id": source_id, "dataset": dataset,
                         "audio": audio, "relative_path": item["relative_path"],
                         "duration_seconds": float(item.get("duration_seconds") or 0),
                         "expected_sha256": item.get("sha256") or "",
                         "usage_tier": item.get("usage_tier"),
                         "audience_label": item.get("audience_label")})
    return rows, summary


def assign_shards(rows: list[dict], count: int) -> list[list[dict]]:
    shards: list[list[dict]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for row in sorted(rows, key=lambda r: (-r["duration_seconds"], r["source_id"])):
        index = min(range(count), key=lambda i: (loads[i], i))
        shards[index].append(row)
        loads[index] += row["duration_seconds"]
    return shards


def audio_duration(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                            capture_output=True, text=True, check=True)
    duration = float(result.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration for {path}")
    return duration


def chunks(duration: float, max_seconds: float):
    for index in range(math.ceil(duration / max_seconds)):
        start = index * max_seconds
        end = min(duration, (index + 1) * max_seconds)
        if end - start > 0.05:
            yield index, start, end


def transcribe_row(row: dict, args: argparse.Namespace, get_transcriber) -> str:
    audio = row["audio"]
    stat = audio.stat()
    key = output_key(row["source_id"])
    chunk_dir = args.out_dir / "chunks" / key
    result_path = args.out_dir / "transcripts" / f"{key}.json"
    config_path = chunk_dir / "config.json"
    config = {"source_id": row["source_id"], "relative_path": row["relative_path"],
              "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
              "expected_sha256": row["expected_sha256"], "model": args.model,
              "compute_type": args.compute_type, "batch_size": args.batch_size,
              "max_clip_sec": args.max_clip_sec, "language": "en"}
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Input or ASR settings changed for {row['source_id']}; use a new output directory")
    else:
        if result_path.exists() or list(chunk_dir.glob("chunk_*.json")):
            raise ValueError(f"Transcript cache lacks config for {row['source_id']}")
        atomic_json(config_path, config)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("source_id") != row["source_id"] or result.get("status") != "complete":
            raise ValueError(f"Invalid existing transcript: {result_path}")
        return "skipped"
    if row["expected_sha256"]:
        digest = hashlib.sha256()
        with audio.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest().lower() != row["expected_sha256"].lower():
            raise ValueError(f"SHA-256 mismatch for {row['source_id']}")
    duration = audio_duration(audio)
    segment_rows = []
    for index, start, end in chunks(duration, args.max_clip_sec):
        cache = chunk_dir / f"chunk_{index:05d}.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            if abs(payload["start_sec"] - start) > 0.02 or abs(payload["end_sec"] - end) > 0.02:
                raise ValueError(f"Chunk boundary changed: {cache}")
        else:
            with tempfile.TemporaryDirectory(prefix="asr_", dir=chunk_dir) as temporary:
                clip = Path(temporary) / "clip.wav"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start),
                                "-i", str(audio), "-t", str(end - start), "-ac", "1",
                                "-ar", "16000", "-vn", str(clip)], check=True)
                try:
                    segments, language = get_transcriber().transcribe_segments(clip)
                except ValueError as exc:
                    if "no timed speech segments" not in str(exc):
                        raise
                    segments, language = [], "en"
            if language != "en":
                raise ValueError(f"Detected {language!r} for English recording {row['source_id']}")
            payload = {"start_sec": start, "end_sec": end, "language": language,
                       "segments": segments}
            atomic_json(cache, payload)
        for segment in payload["segments"]:
            seg_start = start + float(segment["start"])
            seg_end = min(end, start + float(segment["end"]))
            text = " ".join(str(segment["text"]).split())
            if text and 0 <= seg_start < seg_end <= duration + 0.1:
                segment_rows.append({"start_sec": round(seg_start, 3),
                                     "end_sec": round(seg_end, 3), "text": text,
                                     "chunk_index": index})
    if not segment_rows:
        raise ValueError(f"ASR produced no speech for {row['source_id']}")
    atomic_json(result_path, {"schema_version": "asr_transcript_v1", "status": "complete",
                              "source_id": row["source_id"], "dataset": row["dataset"],
                              "relative_path": row["relative_path"],
                              "source_sha256": row["expected_sha256"],
                              "duration_sec": duration, "language": "en",
                              "model": args.model, "compute_type": args.compute_type,
                              "usage_tier": row["usage_tier"],
                              "audience_label": row["audience_label"],
                              "timing_level": "asr_segment_only", "segments": segment_rows})
    return "completed"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/transcription_staged_manifest.jsonl"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--sources", default="LibriVox,YouTube_SciShow_pilot,YouTube_selected_2000",
                        help="Comma-separated datasets to transcribe")
    parser.add_argument("--include-holds", action="store_true", help="Explicitly include exploratory/rights holds")
    parser.add_argument("--out-dir", type=Path, default=Path("results/transcription"))
    parser.add_argument("--shard-count", type=int, default=3)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-clip-sec", type=float, default=25.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-recordings", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("invalid shard index/count")
    if not 0 < args.max_clip_sec <= 30 or args.batch_size < 1 or args.limit_recordings < 0:
        parser.error("clips must be <=30 seconds; batch size must be positive")
    sources = {name.strip() for name in args.sources.split(",") if name.strip()}
    if not sources:
        parser.error("--sources cannot be empty")
    rows, summary = inventory(args.manifest, args.root, sources, args.include_holds)
    unknown = sources - set(summary)
    if unknown:
        parser.error(f"datasets absent from manifest: {sorted(unknown)}")
    shards = assign_shards(rows, args.shard_count)
    selected = shards[args.shard_index]
    if args.limit_recordings:
        selected = selected[:args.limit_recordings]
    report = {"inventory": summary, "selected_sources": sorted(sources),
              "shard_count": args.shard_count, "shard_index": args.shard_index,
              "shard_recordings": len(selected),
              "shard_hours": round(sum(r["duration_seconds"] for r in selected) / 3600, 3),
              "source_ids": [r["source_id"] for r in selected]}
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this worker")
    transcriber = None

    def get_transcriber():
        nonlocal transcriber
        if transcriber is None:
            from speech_pipeline import GPUTranscriber
            transcriber = GPUTranscriber(args.device, model_size=args.model,
                                         compute_type=args.compute_type,
                                         batch_size=args.batch_size, language="en")
        return transcriber

    counts = {"completed": 0, "skipped": 0, "failed": 0}
    for row in selected:
        try:
            outcome = transcribe_row(row, args, get_transcriber)
            counts[outcome] += 1
            LOG.info("%s %s", outcome, row["source_id"])
        except Exception as exc:
            counts["failed"] += 1
            LOG.exception("Failed %s: %s", row["source_id"], exc)
    print(json.dumps({**report, "counts": counts}, indent=2))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
