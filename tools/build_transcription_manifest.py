"""Combine the staged speech manifest with locally present selected YouTube audio."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


AUDIO_EXTENSIONS = {".m4a", ".webm", ".opus", ".mp3", ".ogg", ".wav", ".flac"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_rows(base_manifest: Path, selection_csv: Path, audio_root: Path, root: Path) -> tuple[list[dict], dict]:
    rows = [json.loads(line) for line in base_manifest.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    with selection_csv.open(newline="", encoding="utf-8-sig") as stream:
        candidates = list(csv.DictReader(stream))
    by_id: dict[str, list[dict]] = {}
    for candidate in candidates:
        video_id = candidate.get("video_id", "")
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            by_id.setdefault(video_id, []).append(candidate)
    counts = {"url_candidates": len(candidates), "unique_ids": len(by_id),
              "included_audio": 0, "known_non_english": 0,
              "ambiguous_audio": 0, "missing_audio": 0, "missing_metadata": 0}
    known_ids = {row["source_id"] for row in rows}
    for video_id, candidates_for_id in sorted(by_id.items()):
        matches = [path for path in audio_root.glob(f"{video_id}.*")
                   if path.suffix.lower() in AUDIO_EXTENSIONS and path.stat().st_size > 0]
        if not matches:
            counts["missing_audio"] += 1
            continue
        if len(matches) != 1:
            counts["ambiguous_audio"] += 1
            continue
        info_path = audio_root / f"{video_id}.info.json"
        if not info_path.is_file():
            counts["missing_metadata"] += 1
            continue
        info = json.loads(info_path.read_text(encoding="utf-8"))
        language = info.get("language")
        if language and not str(language).lower().startswith("en"):
            counts["known_non_english"] += 1
            continue
        duration = float(info.get("duration") or 0)
        if duration <= 0:
            counts["missing_metadata"] += 1
            continue
        audio = matches[0]
        source_id = f"YouTube:{video_id}"
        if source_id in known_ids:
            raise ValueError(f"Duplicate source ID: {source_id}")
        known_ids.add(source_id)
        kinds = sorted({str(candidate.get("kind") or "").strip()
                        for candidate in candidates_for_id if candidate.get("kind")})
        rows.append({"source_id": source_id, "dataset": "YouTube_selected_2000",
                     "recording_id": video_id, "relative_path": audio.relative_to(root).as_posix(),
                     "acquisition_status": "staged", "transcript_status": "whisperx_required",
                     "usage_tier": "exploratory_hold", "rights_review": "hold",
                     "language": "en", "language_basis": "metadata" if language else "assumed_for_asr_review",
                     "duration_seconds": duration, "bytes": audio.stat().st_size,
                     "sha256": sha256(audio), "candidate_kinds": kinds,
                     "audience_label": None, "audience_label_basis": "selection_candidate_only",
                     "title": info.get("title"),
                     "source_page_url": f"https://www.youtube.com/watch?v={video_id}"})
        counts["included_audio"] += 1
    return rows, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--base-manifest", type=Path,
                        default=Path("artifacts/weekend_staged_manifest.jsonl"))
    parser.add_argument("--selection-csv", type=Path,
                        default=Path("artifacts/videos_selected_all_2000.csv"))
    parser.add_argument("--audio-root", type=Path,
                        default=Path("data/staging/youtube_selected_2000/audio"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/transcription_staged_manifest.jsonl"))
    args = parser.parse_args()
    root = args.root.resolve()
    rows, counts = build_rows(root / args.base_manifest, root / args.selection_csv,
                              root / args.audio_root, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                           encoding="utf-8")
    print(json.dumps(counts, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
