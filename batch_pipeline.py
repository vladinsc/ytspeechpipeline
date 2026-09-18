"""Resumable batch runner for the CDS prosody extraction pipeline.

The URL file contains one public YouTube URL per line under ``# LABEL:`` section
markers. A JSON checkpoint is atomically updated at every major pipeline stage,
allowing both live inspection and restart after interruption.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from speech_pipeline import (
    GPUTranscriber,
    PipelineConfig,
    ProsodyPipeline,
    VocalIsolator,
    resolve_device,
)

log = logging.getLogger("batch")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def nvidia_smi_snapshot() -> dict:
    """Return a compact, checkpoint-friendly snapshot of visible NVIDIA GPUs."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if process.returncode != 0:
        return {"available": False, "error": process.stderr.strip() or "nvidia-smi failed"}

    fields = ["index", "name", "uuid", "driver_version", "memory_total_mib",
              "memory_used_mib", "utilization_percent"]
    devices = []
    for line in process.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            devices.append(dict(zip(fields, values)))
    return {"available": bool(devices), "captured_at": utc_now(), "devices": devices}


LABELS = {
    "YOUTUBE_KIDS": "kids",
    "KIDS": "kids",
    "NORMAL_YOUTUBE": "normal",
    "YOUTUBE_NORMAL": "normal",
    "NORMAL": "normal",
}


def load_manifest(path: Path) -> list[dict[str, str]]:
    videos: list[dict[str, str]] = []
    seen: dict[str, str] = {}
    current_label: Optional[str] = None
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        label_match = re.match(r"^#\s*LABEL:\s*([A-Z_]+)", line, flags=re.IGNORECASE)
        if label_match:
            raw_label = label_match.group(1).upper()
            if raw_label not in LABELS:
                raise ValueError(f"Unknown label {raw_label!r} in {path}")
            current_label = LABELS[raw_label]
            continue
        if not line or line.startswith("#"):
            continue
        url = line
        if not url.startswith(("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")):
            raise ValueError(f"Unsupported URL in {path}: {url}")
        if url in seen:
            if seen[url] != current_label:
                raise ValueError(f"URL appears under two labels ({seen[url]}, {current_label}): {url}")
            log.warning("Skipping duplicate URL: %s", url)
            continue
        if current_label is None:
            raise ValueError(f"URL appears before a # LABEL: section in {path}: {url}")
        seen[url] = current_label
        videos.append({"url": url, "label": current_label})
    if not videos:
        raise ValueError(f"No YouTube URLs found in {path}")
    return videos


def load_urls(path: Path) -> list[str]:
    """Compatibility helper returning only URLs from a labeled manifest."""
    return [video["url"] for video in load_manifest(path)]


def shard_manifest(
    videos: list[dict[str, str]], shard_count: int, shard_index: int
) -> list[dict]:
    """Select one deterministic shard while retaining global manifest indexes."""
    if shard_count < 1:
        raise ValueError("--shard-count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"--shard-index must be between 0 and {shard_count - 1} "
            f"for --shard-count {shard_count}"
        )

    selected: list[dict] = []
    for manifest_index, video in enumerate(videos, start=1):
        if (manifest_index - 1) % shard_count == shard_index:
            selected.append({**video, "manifest_index": manifest_index})
    return selected


def video_label(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc.endswith("youtu.be"):
        candidate = parsed.path.strip("/").split("/")[0]
    else:
        candidate = parse_qs(parsed.query).get("v", [""])[0]
    candidate = re.sub(r"[^A-Za-z0-9_-]", "", candidate)
    return candidate[:32] or "video"


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def new_entry(index: int, video: dict[str, str], output_dir: Path, work_root: Path) -> dict:
    url, label = video["url"], video["label"]
    stem = f"video_{index:04d}_{video_label(url)}"
    return {
        "index": index,
        "url": url,
        "label": label,
        "status": "pending",
        "stage": "waiting",
        "stage_status": "pending",
        "detail": "",
        "attempts": 0,
        "output_path": str((output_dir / label / f"{stem}.json").resolve()),
        "workdir": str((work_root / label / stem).resolve()),
        "started_at": None,
        "completed_at": None,
        "updated_at": utc_now(),
        "record_counts": None,
        "error": None,
    }


def load_or_create_checkpoint(
    path: Path, videos: list[dict[str, str]], urls_file: Path, output_dir: Path, work_root: Path
) -> dict:
    old_entries: dict[str, dict] = {}
    created_at = utc_now()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old_entries = {entry["url"]: entry for entry in existing.get("videos", [])}
        created_at = existing.get("created_at", created_at)

    entries: list[dict] = []
    for position, video in enumerate(videos, start=1):
        index = int(video.get("manifest_index", position))
        url = video["url"]
        fresh = new_entry(index, video, output_dir, work_root)
        previous = old_entries.get(url)
        if previous:
            # Output/work paths follow the current CLI arguments, while status is resumed.
            fresh.update({key: value for key, value in previous.items()
                          if key not in {"index", "label", "output_path", "workdir"}})
            fresh["index"] = index
            if fresh["status"] in {"processing", "interrupted"}:
                fresh["status"] = "pending"
                fresh["stage_status"] = "interrupted"
                fresh["detail"] = "Previous process stopped; queued for automatic retry."
        entries.append(fresh)

    return {
        "schema_version": 1,
        "created_at": created_at,
        "updated_at": utc_now(),
        "urls_file": str(urls_file.resolve()),
        "output_dir": str(output_dir.resolve()),
        "work_root": str(work_root.resolve()),
        "model_status": "not_loaded",
        "current_video": None,
        "summary": {},
        "videos": entries,
    }


def update_summary(checkpoint: dict) -> None:
    entries = checkpoint["videos"]
    checkpoint["updated_at"] = utc_now()
    checkpoint["summary"] = {
        "total": len(entries),
        "pending": sum(entry["status"] == "pending" for entry in entries),
        "processing": sum(entry["status"] == "processing" for entry in entries),
        "completed": sum(entry["status"] == "completed" for entry in entries),
        "failed": sum(entry["status"] == "failed" for entry in entries),
        "interrupted": sum(entry["status"] == "interrupted" for entry in entries),
        "by_label": {
            label: {
                "total": sum(entry["label"] == label for entry in entries),
                "completed": sum(entry["label"] == label and entry["status"] == "completed"
                                 for entry in entries),
                "failed": sum(entry["label"] == label and entry["status"] == "failed"
                              for entry in entries),
            }
            for label in sorted({entry["label"] for entry in entries})
        },
    }


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    update_summary(checkpoint)
    atomic_write_json(path, checkpoint)


def count_records(payload: list[dict] | dict[str, list[dict]]) -> dict[str, int]:
    if isinstance(payload, list):
        return {"records": len(payload)}
    return {name: len(records) for name, records in payload.items() if isinstance(records, list)}


def default_checkpoint_path(output_dir: Path, shard_count: int, shard_index: int) -> Path:
    if shard_count == 1:
        return output_dir / "batch_checkpoint.json"
    return output_dir / f"batch_checkpoint_shard_{shard_index}_of_{shard_count}.json"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a resumable list of public YouTube videos.")
    parser.add_argument("urls_file", type=Path, help="Text file containing one YouTube URL per line.")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"))
    parser.add_argument("--work-root", type=Path, default=Path("./_batch_work"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint JSON path (default: a shard-specific file in OUTPUT_DIR).")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Number of parallel manifest shards (default: 1).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based shard handled by this process (default: 0).")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=[1],
        default=1,
        help="WhisperX transcription batch size (fixed at 1 to limit GPU memory use).",
    )
    parser.add_argument("--demucs-model", default="htdemucs")
    parser.add_argument(
        "--demucs-device", default=None,
        help="Device for Demucs (default: same as --device; e.g. cuda:1).",
    )
    parser.add_argument(
        "--alignment-device", default=None,
        help="Device for forced alignment (default: same as --device; e.g. cuda:1).",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--pitch-floor", type=float, default=75.0)
    parser.add_argument("--pitch-ceiling", type=float, default=600.0)
    parser.add_argument("--granularity", choices=["word", "syllable", "both"], default="both")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Retry entries marked failed in an existing checkpoint.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--keep-workdirs", action="store_true",
                        help="Keep intermediate audio even after successful videos.")
    parser.add_argument("--delete-failed-workdirs", action="store_true",
                        help="Also delete partial audio after a failed video.")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    checkpoint_path = args.checkpoint or default_checkpoint_path(
        args.output_dir, args.shard_count, args.shard_index
    )
    try:
        all_videos = load_manifest(args.urls_file)
        videos = shard_manifest(all_videos, args.shard_count, args.shard_index)
        checkpoint = load_or_create_checkpoint(
            checkpoint_path, videos, args.urls_file, args.output_dir, args.work_root
        )
        checkpoint["shard"] = {
            "index": args.shard_index,
            "count": args.shard_count,
            "videos": len(videos),
            "manifest_videos": len(all_videos),
        }
    except Exception as exc:
        log.error("Cannot initialize batch: %s", exc)
        return 2

    log.info(
        "Manifest shard %d/%d contains %d of %d videos",
        args.shard_index + 1, args.shard_count, len(videos), len(all_videos),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    if args.retry_failed:
        for entry in checkpoint["videos"]:
            if entry["status"] == "failed":
                entry["status"] = "pending"
                entry["stage"] = "waiting"
                entry["stage_status"] = "retry_queued"
                entry["error"] = None

    # A completed entry is only trustworthy while its JSON still exists.
    for entry in checkpoint["videos"]:
        if entry["status"] == "completed" and not Path(entry["output_path"]).exists():
            entry["status"] = "pending"
            entry["stage_status"] = "output_missing"
    save_checkpoint(checkpoint_path, checkpoint)

    pending = [entry for entry in checkpoint["videos"] if entry["status"] == "pending"]
    if not pending:
        log.info("Nothing to process. Summary: %s", checkpoint["summary"])
        return 1 if checkpoint["summary"]["failed"] else 0

    try:
        checkpoint["model_status"] = "loading"
        save_checkpoint(checkpoint_path, checkpoint)
        device = resolve_device(args.device)
        demucs_device = args.demucs_device or device
        alignment_device = args.alignment_device or device
        if any(str(value).startswith("cuda") for value in
               (device, demucs_device, alignment_device)):
            checkpoint["gpu"] = nvidia_smi_snapshot()
            if not checkpoint["gpu"]["available"]:
                raise RuntimeError(f"CUDA selected but nvidia-smi failed: {checkpoint['gpu'].get('error')}")
            for gpu in checkpoint["gpu"]["devices"]:
                log.info(
                    "NVIDIA GPU %s: %s | driver=%s | memory=%s MiB",
                    gpu["index"], gpu["name"], gpu["driver_version"], gpu["memory_total_mib"],
                )
            save_checkpoint(checkpoint_path, checkpoint)
        log.info(
            "Stage devices: WhisperX=%s, Demucs=%s, alignment=%s, Silero/Praat=cpu",
            device, demucs_device, alignment_device,
        )
        isolator = VocalIsolator(demucs_device, model_name=args.demucs_model)
        transcriber = GPUTranscriber(
            device,
            model_size=args.whisper_model,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            language=args.language,
            alignment_device=alignment_device,
        )
        checkpoint["model_status"] = "loaded"
        checkpoint["device"] = device
        checkpoint["devices"] = {
            "whisperx": device,
            "demucs": demucs_device,
            "alignment": alignment_device,
            "silero": "cpu",
            "praat": "cpu",
        }
        save_checkpoint(checkpoint_path, checkpoint)
    except Exception as exc:
        checkpoint["model_status"] = "load_failed"
        checkpoint["model_error"] = f"{type(exc).__name__}: {exc}"
        save_checkpoint(checkpoint_path, checkpoint)
        log.exception("Could not load pipeline models")
        return 2

    total = len(checkpoint["videos"])
    for entry in checkpoint["videos"]:
        if entry["status"] != "pending":
            continue

        index = entry["index"]
        log.info("[%d/%d] Starting %s", index, total, entry["url"])
        entry.update({
            "status": "processing",
            "stage": "initializing",
            "stage_status": "started",
            "detail": "",
            "attempts": entry.get("attempts", 0) + 1,
            "started_at": utc_now(),
            "completed_at": None,
            "updated_at": utc_now(),
            "error": None,
        })
        checkpoint["current_video"] = index
        save_checkpoint(checkpoint_path, checkpoint)

        def on_progress(stage: str, status: str, detail: str) -> None:
            entry["stage"] = stage
            entry["stage_status"] = status
            entry["detail"] = detail
            entry["updated_at"] = utc_now()
            save_checkpoint(checkpoint_path, checkpoint)

        cfg = PipelineConfig(
            url=entry["url"],
            out_path=Path(entry["output_path"]),
            workdir=Path(entry["workdir"]),
            device=device,
            demucs_device=demucs_device,
            alignment_device=alignment_device,
            whisper_model=args.whisper_model,
            compute_type=args.compute_type,
            demucs_model=args.demucs_model,
            language=args.language,
            pitch_floor=args.pitch_floor,
            pitch_ceiling=args.pitch_ceiling,
            keep_workdir=args.keep_workdirs,
            granularity=args.granularity,
            label=entry["label"],
        )
        try:
            payload = ProsodyPipeline(cfg, device=device).run(
                isolator=isolator,
                transcriber=transcriber,
                progress_callback=on_progress,
            )
            if not cfg.out_path.exists() or cfg.out_path.stat().st_size == 0:
                raise RuntimeError("Pipeline returned without a non-empty output JSON")
            entry["status"] = "completed"
            entry["stage"] = "finished"
            entry["stage_status"] = "completed"
            entry["detail"] = str(cfg.out_path)
            entry["record_counts"] = count_records(payload)
            entry["completed_at"] = utc_now()
            log.info("[%d/%d] Completed -> %s", index, total, cfg.out_path)
        except KeyboardInterrupt:
            entry["status"] = "interrupted"
            entry["stage_status"] = "interrupted"
            entry["error"] = "KeyboardInterrupt"
            checkpoint["current_video"] = None
            save_checkpoint(checkpoint_path, checkpoint)
            log.warning("Batch interrupted; resume with the same command.")
            return 130
        except Exception as exc:
            entry["status"] = "failed"
            entry["stage_status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            log.exception("[%d/%d] Failed %s", index, total, entry["url"])
            if args.delete_failed_workdirs:
                shutil.rmtree(cfg.workdir, ignore_errors=True)
            if args.stop_on_error:
                checkpoint["current_video"] = None
                save_checkpoint(checkpoint_path, checkpoint)
                return 1
        finally:
            entry["updated_at"] = utc_now()
            checkpoint["current_video"] = None
            save_checkpoint(checkpoint_path, checkpoint)

    log.info("Batch finished. Summary: %s", checkpoint["summary"])
    return 1 if checkpoint["summary"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
