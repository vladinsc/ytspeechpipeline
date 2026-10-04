"""Prepare or execute the two-per-source server validation and video bundle.

Default mode checks the selected files only. --execute requires Linux/CUDA,
runs the feature pipeline, and renders one annotated video per source.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from render_validation_video import MAX_SECONDS, render


DATASETS = {
    "HVC", "LibriVox", "AMI", "NewmanRatner", "VoxPopuli",
    "YouTube_SciShow_pilot", "YouTube_selected_2000",
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def _plan(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != "server_validation_plan_v1":
        raise ValueError("Unsupported validation plan")
    if not 0 < float(data.get("video_max_sec", 0)) <= MAX_SECONDS:
        raise ValueError("Video trim must be at most 600 seconds")
    examples = data["examples"]
    counts = Counter(item["dataset"] for item in examples)
    if counts != Counter({name: 2 for name in DATASETS}):
        raise ValueError(f"Expected exactly two examples for each of seven sources: {counts}")
    source_ids = [item["source_id"] for item in examples]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Duplicate source_id in validation plan")
    return examples


def _check_sources(args, examples: list[dict]) -> list[dict]:
    if str(args.root) not in sys.path:
        sys.path.insert(0, str(args.root))
    from run_feature_batch import build_queue

    queue_args = argparse.Namespace(
        root=args.root, manifest=args.manifest,
        newman_root=args.newman_root, vox_manifest=args.vox_manifest,
        youtube_manifest=args.youtube_manifest, youtube_root=args.youtube_root,
        sources=",".join(sorted(DATASETS)),
    )
    available = {row["source_id"]: row for row in build_queue(queue_args)}
    checked = []
    for item in examples:
        row = available.get(item["source_id"])
        if row is None:
            raise ValueError(f"Selected source is unavailable: {item['source_id']}")
        if row["dataset"] != item["dataset"]:
            raise ValueError(f"Dataset mismatch for {item['source_id']}")
        checked.append({"dataset": item["dataset"], "source_id": item["source_id"],
                        "stratum": item["stratum"], "audio_path": str(row["_audio"])})
    return checked


def _read_features(path: Path, expected: set[str]) -> dict[str, list[dict]]:
    by_source: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row["source_id"] in expected:
                    by_source[row["source_id"]].append(row)
    return by_source


def _assert_feature_coverage(by_source: dict[str, list[dict]], examples: list[dict]) -> None:
    for item in examples:
        source_id = item["source_id"]
        records = by_source.get(source_id, [])
        if not records:
            raise RuntimeError(f"Feature pipeline produced no rows for {source_id}")
        if item["dataset"] != "HVC" and not any(
            row.get("transcript") and row.get("words") and row.get("phones")
            for row in records
        ):
            raise RuntimeError(f"No transcribed and aligned utterance for {source_id}")
        if not any(row.get("acoustic_features") for row in records):
            raise RuntimeError(f"No acoustic features for {source_id}")


def _run_pipeline(args, examples: list[dict]) -> Path:
    output = args.output / "feature_run"
    command = [
        sys.executable, str(args.root / "run_feature_batch.py"),
        "--root", str(args.root), "--manifest", str(args.manifest),
        "--newman-root", str(args.newman_root),
        "--vox-manifest", str(args.vox_manifest),
        "--youtube-manifest", str(args.youtube_manifest),
        "--youtube-root", str(args.youtube_root),
        "--sources", ",".join(sorted(DATASETS)),
        "--out-dir", str(output),
        "--mfa-conda-env", args.mfa_conda_env,
        "--asr-device", "cuda",
        "--stop-on-error",
    ]
    if args.speaker_segments:
        command.extend(["--speaker-segments", str(args.speaker_segments)])
    for item in examples:
        command.extend(["--source-id", item["source_id"]])
    subprocess.run(command, cwd=args.root, check=True)
    return output / "features.jsonl"


def execute(args, examples: list[dict]) -> dict:
    if platform.system() != "Linux":
        raise RuntimeError("Execute this validation only on the Linux GPU server")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; start the GPU server first")
    features = _run_pipeline(args, examples)
    by_source = _read_features(features, {item["source_id"] for item in examples})
    _assert_feature_coverage(by_source, examples)
    videos = args.output / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    report = []
    for index, item in enumerate(examples, 1):
        source_id = item["source_id"]
        records = by_source[source_id]
        output = videos / f"{index:02d}_{_safe_name(source_id)}.mp4"
        video = render(source_id, records, output, args.max_video_sec)
        flags = Counter(flag for row in records for flag in row.get("quality_flags", []))
        report.append({
            **item, **video,
            "feature_rows": len(records),
            "speaker_ids": sorted({row["speaker_id"] for row in records if row.get("speaker_id")}),
            "transcript_sources": sorted({row["transcript_source"] for row in records}),
            "quality_flags": dict(flags),
            "alignment_word_count_mismatches": flags.get("alignment_word_count_mismatch", 0),
        })
        print(f"Rendered {index}/{len(examples)}: {output}", flush=True)
    result = {"status": "complete", "sources": len(report),
              "feature_jsonl": str(features), "examples": report}
    (args.output / "validation_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path,
                        default=Path("artifacts/server_validation_plan.json"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path,
                        default=Path("artifacts/server_validation_staged_manifest.jsonl"))
    parser.add_argument("--newman-root", type=Path,
                        default=Path("data/staging/newman_ratner"))
    parser.add_argument("--vox-manifest", type=Path,
                        default=Path("artifacts/validation_vox_2_manifest.jsonl"))
    parser.add_argument("--youtube-manifest", type=Path,
                        default=Path("artifacts/server_validation_selected_youtube.csv"))
    parser.add_argument("--youtube-root", type=Path,
                        default=Path("data/staging/youtube_selected_2000/audio"))
    parser.add_argument("--speaker-segments", type=Path)
    parser.add_argument("--mfa-conda-env", default="aligner")
    parser.add_argument("--output", type=Path,
                        default=Path("results/server_validation"))
    parser.add_argument("--max-video-sec", type=float, default=MAX_SECONDS)
    parser.add_argument("--execute", action="store_true",
                        help="Run GPU features and videos on the Linux server")
    args = parser.parse_args()
    args.root = args.root.resolve()
    examples = _plan(args.plan)
    checked = _check_sources(args, examples)
    if not 0 < args.max_video_sec <= MAX_SECONDS:
        parser.error("--max-video-sec must be between 0 and 600")
    if not args.execute:
        print(json.dumps({"status": "ready", "examples": checked}, indent=2))
        return 0
    result = execute(args, examples)
    print(json.dumps({"status": result["status"], "sources": result["sources"],
                      "report": str(args.output / "validation_report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
