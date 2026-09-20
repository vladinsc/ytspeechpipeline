"""Build a diverse, metadata-verified YouTube candidate manifest.

This script never downloads media.  It first discovers public English-targeted
videos using yt-dlp search and then performs yt-dlp metadata/format extraction
for every retained URL.  A row passes only if an unauthenticated extraction
exposes at least one direct audio-bearing format.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from yt_dlp import YoutubeDL

MIN_SECONDS = 5 * 60
MAX_SECONDS = 10 * 60
YEARS = list(range(2008, datetime.now(UTC).year + 1))

# Topic diversity is intentional. Each phrase is searched with each year.
KIDS_TOPICS = [
    ("cartoon", "kids cartoon full episode English"),
    ("music", "nursery rhymes songs English kids"),
    ("story", "children storytime read aloud English"),
    ("science", "kids science experiment English"),
    ("nature", "kids wildlife nature video English"),
    ("space", "kids space science English"),
    ("history", "kids history lesson English"),
    ("geography", "kids geography lesson English"),
    ("math", "kids maths learning English"),
    ("coding", "kids coding tutorial English"),
    ("art", "kids drawing art tutorial English"),
    ("craft", "kids crafts tutorial English"),
    ("cooking", "kids cooking video English"),
    ("movement", "kids dance movement yoga English"),
    ("music_lesson", "kids music lesson English"),
    ("reading", "phonics reading lesson English kids"),
    ("engineering", "kids engineering science English"),
    ("shows", "PBS Kids English full episode"),
    ("educational", "educational video for kids English"),
]

NORMAL_TOPICS = [
    ("vlog_daily", "English daily vlog"),
    ("vlog_travel", "English travel vlog"),
    ("vlog_study", "English study vlog"),
    ("vlog_cooking", "English cooking vlog"),
    ("vlog_fitness", "English fitness vlog"),
    ("documentary_history", "short history documentary English"),
    ("documentary_nature", "short nature documentary English"),
    ("documentary_science", "short science documentary English"),
    ("documentary_culture", "short culture documentary English"),
    ("interview", "English interview conversation"),
    ("podcast", "English podcast conversation"),
    ("technology", "technology explainer English"),
    ("software", "software tutorial English"),
    ("education", "educational explainer English"),
    ("news", "news explainer English"),
    ("comedy", "English comedy sketch"),
    ("sports", "sports analysis English"),
    ("music", "music lesson English"),
    ("diy", "DIY tutorial English"),
    ("gaming", "English gaming commentary"),
    ("lifestyle", "English lifestyle video"),
    ("business", "business explainer English"),
]


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def discover(ydl: YoutubeDL, topics: list[tuple[str, str]], target: int, source: str) -> list[dict]:
    """Round-robin topic/year discovery, with enough surplus for failed checks."""
    wanted = int(target * 1.8)
    rows: list[dict] = []
    seen: set[str] = set()
    for year in YEARS:
        for genre, phrase in topics:
            query = f"{phrase} {year}"
            try:
                result = ydl.extract_info(f"ytsearch25:{query}", download=False)
            except Exception as exc:
                print(f"WARN search failed: {query!r}: {exc}")
                continue
            for item in (result or {}).get("entries", []):
                if not item:
                    continue
                video_id = clean(item.get("id"))
                duration = item.get("duration")
                if video_id in seen or not isinstance(duration, (int, float)):
                    continue
                if not MIN_SECONDS <= duration <= MAX_SECONDS:
                    continue
                seen.add(video_id)
                rows.append({
                    "source": source,
                    "classification": "kids_candidate" if source == "youtube_kids" else "normal_youtube",
                    "genre": genre,
                    "title": clean(item.get("title")),
                    "channel": clean(item.get("channel") or item.get("uploader")),
                    "duration_seconds": int(duration),
                    "video_id": video_id,
                    "url": item.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
                    "kids_url": f"https://www.youtubekids.com/watch/{video_id}" if source == "youtube_kids" else "",
                    "search_query": query,
                    "search_year": year,
                })
                if len(rows) >= wanted:
                    return rows
    return rows


def verify(candidate: dict) -> dict | None:
    """Resolve formats without downloading media or supplying authentication."""
    options = {"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(candidate["url"], download=False)
    except Exception:
        return None
    if not info or info.get("availability") in {"private", "premium_only", "subscriber_only"}:
        return None
    if not any(f.get("url") and f.get("acodec") not in (None, "none") for f in info.get("formats", [])):
        return None
    language = clean(info.get("language")).lower()
    # Metadata language is often absent. Search terms select English; explicit
    # non-English metadata is rejected.
    if language and not language.startswith("en"):
        return None
    duration = info.get("duration")
    if not isinstance(duration, (int, float)) or not MIN_SECONDS <= duration <= MAX_SECONDS:
        return None
    return {
        **candidate,
        "title": clean(info.get("title")) or candidate["title"],
        "channel": clean(info.get("channel") or info.get("uploader")) or candidate["channel"],
        "duration_seconds": int(duration),
        "language": language or "english_search_candidate",
        "download_check": "passed_unauthenticated_metadata_format_check",
        "download_checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def verify_all(candidates: list[dict], target: int, label: str) -> list[dict]:
    accepted: list[dict] = []
    for checked, candidate in enumerate(candidates, 1):
        row = verify(candidate)
        if row:
            accepted.append(row)
        if checked % 100 == 0:
            print(f"{label}: checked {checked}, accepted {len(accepted)}/{target}")
            time.sleep(1)
        if len(accepted) == target:
            break
    return accepted


def write_outputs(output: Path, kids: list[dict], normal: list[dict]) -> None:
    rows = kids + normal
    fields = list(rows[0]) if rows else ["source", "url"]
    with (output / "youtube_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "youtube_manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    header = "# Metadata collected by build_verified_video_manifest.py; media not downloaded\n"
    (output / "videos_kids.txt").write_text(header + "\n".join(row["url"] for row in kids) + "\n", encoding="utf-8")
    (output / "videos_normal.txt").write_text(header + "\n".join(row["url"] for row in normal) + "\n", encoding="utf-8")
    (output / "videos.txt").write_text(
        header + "# LABEL: YOUTUBE_KIDS\n" + "\n".join(row["url"] for row in kids)
        + "\n# LABEL: NORMAL_YOUTUBE\n" + "\n".join(row["url"] for row in normal) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-kids", type=int, default=2500)
    parser.add_argument("--target-normal", type=int, default=2500)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    search_options = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with YoutubeDL(search_options) as ydl:
        kids_candidates = discover(ydl, KIDS_TOPICS, args.target_kids, "youtube_kids")
        normal_candidates = discover(ydl, NORMAL_TOPICS, args.target_normal, "youtube")
    print(f"Candidates: kids={len(kids_candidates)}, normal={len(normal_candidates)}")
    kids = verify_all(kids_candidates, args.target_kids, "kids")
    normal = verify_all(normal_candidates, args.target_normal, "normal")
    write_outputs(args.out_dir, kids, normal)
    print(f"Verified: kids={len(kids)}/{args.target_kids}, normal={len(normal)}/{args.target_normal}")
    return 0 if len(kids) == args.target_kids and len(normal) == args.target_normal else 2


if __name__ == "__main__":
    raise SystemExit(main())
