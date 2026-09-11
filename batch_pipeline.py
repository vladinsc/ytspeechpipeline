"""Resumable batch runner for the CDS prosody extraction pipeline.

The URL file contains one public YouTube URL per line. Empty lines and lines
starting with ``#`` are ignored. A JSON checkpoint is atomically updated at every
major pipeline stage, allowing both live inspection and restart after interruption.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
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


def load_urls(path: Path) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        url = raw_line.strip()
        if not url or url.startswith("#"):
            continue
        if not url.startswith(("https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/")):
            raise ValueError(f"Unsupported URL in {path}: {url}")
        if url in seen:
            log.warning("Skipping duplicate URL: %s", url)
            continue
        seen.add(url)
        urls.append(url)
    if not urls:
        raise ValueError(f"No YouTube URLs found in {path}")
    return urls


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


def new_entry(index: int, url: str, output_dir: Path, work_root: Path) -> dict:
    stem = f"video_{index:04d}_{video_label(url)}"
    return {
        "index": index,
        "url": url,
        "status": "pending",
        "stage": "waiting",
        "stage_status": "pending",
        "detail": "",
        "attempts": 0,
        "output_path": str((output_dir / f"{stem}.json").resolve()),
        "workdir": str((work_root / stem).resolve()),
        "started_at": None,
        "completed_at": None,
        "updated_at": utc_now(),
        "record_counts": None,
        "error": None,
    }


def load_or_create_checkpoint(
    path: Path, urls: list[str], urls_file: Path, output_dir: Path, work_root: Path
) -> dict:
    old_entries: dict[str, dict] = {}
    created_at = utc_now()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old_entries = {entry["url"]: entry for entry in existing.get("videos", [])}
        created_at = existing.get("created_at", created_at)

    entries: list[dict] = []
    for index, url in enumerate(urls, start=1):
        fresh = new_entry(index, url, output_dir, work_root)
        previous = old_entries.get(url)
        if previous:
            # Output/work paths follow the current CLI arguments, while status is resumed.
            fresh.update({key: value for key, value in previous.items()
                          if key not in {"index", "output_path", "workdir"}})
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
    }


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    update_summary(checkpoint)
    atomic_write_json(path, checkpoint)


def count_records(payload: list[dict] | dict[str, list[dict]]) -> dict[str, int]:
    if isinstance(payload, list):
        return {"records": len(payload)}
    return {name: len(records) for name, records in payload.items()}


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a resumable list of public YouTube videos.")
    parser.add_argument("urls_file", type=Path, help="Text file containing one YouTube URL per line.")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"))
    parser.add_argument("--work-root", type=Path, default=Path("./_batch_work"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Checkpoint JSON path (default: OUTPUT_DIR/batch_checkpoint.json).")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--demucs-model", default="htdemucs")
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
    checkpoint_path = args.checkpoint or args.output_dir / "batch_checkpoint.json"
    try:
        urls = load_urls(args.urls_file)
        checkpoint = load_or_create_checkpoint(
            checkpoint_path, urls, args.urls_file, args.output_dir, args.work_root
        )
    except Exception as exc:
        log.error("Cannot initialize batch: %s", exc)
        return 2

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
        isolator = VocalIsolator(device, model_name=args.demucs_model)
        transcriber = GPUTranscriber(
            device,
            model_size=args.whisper_model,
            compute_type=args.compute_type,
            language=args.language,
        )
        checkpoint["model_status"] = "loaded"
        checkpoint["device"] = device
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
            whisper_model=args.whisper_model,
            compute_type=args.compute_type,
            demucs_model=args.demucs_model,
            language=args.language,
            pitch_floor=args.pitch_floor,
            pitch_ceiling=args.pitch_ceiling,
            keep_workdir=args.keep_workdirs,
            granularity=args.granularity,
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
