"""Collect a metadata-only English YouTube/YouTube Kids candidate manifest.

This intentionally does not download media.  YouTube Kids uses the same public
video IDs as YouTube; the Kids set is constructed from child-directed channels
and queries and is labelled as ``kids_candidate`` for later human review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from yt_dlp import YoutubeDL

MIN_SECONDS = 5 * 60
MAX_SECONDS = 10 * 60

KIDS_QUERIES = [
    ("early_childhood", "Super Simple Songs English 5 minutes"),
    ("early_childhood", "Cocomelon English song 5 minutes"),
    ("early_childhood", "Sesame Street English 5 minutes"),
    ("early_childhood", "PBS Kids story English 5 minutes"),
    ("early_childhood", "Blippi educational English 5 minutes"),
    ("early_childhood", "Peppa Pig English full episode 5 minutes"),
    ("school_age", "SciShow Kids English 5 minutes"),
    ("school_age", "National Geographic Kids English 5 minutes"),
    ("school_age", "Storyline Online read aloud English 5 minutes"),
    ("school_age", "Art for Kids Hub drawing tutorial 5 minutes"),
    ("school_age", "Cosmic Kids Yoga English 5 minutes"),
    ("school_age", "Numberblocks English 5 minutes"),
    ("school_age", "Wild Kratts PBS Kids 5 minutes"),
    ("school_age", "Kids learning science English 5 minutes"),
    ("tweens", "Crash Course Kids English 5 minutes"),
    ("tweens", "TED-Ed kids science English 5 minutes"),
    ("tweens", "Mark Rober kids science 8 minutes"),
    ("tweens", "BrainPOP English 5 minutes"),
    ("school_age", "The Dad Lab science experiment kids English 6 minutes"),
    ("school_age", "Homeschool Pop English 5 minutes"),
    ("early_childhood", "Little Baby Bum English 5 minutes"),
    ("tweens", "Mocomi kids educational English 6 minutes"),
    ("school_age", "MinuteEarth kids English 6 minutes"),
    ("school_age", "Easy Kids Crafts English 6 minutes"),
    ("early_childhood", "Mother Goose Club English 5 minutes"),
    ("school_age", "kids educational video English 5 min learning"),
    ("school_age", "fun facts for kids English 5 minutes"),
    ("tweens", "kids history lesson English 6 minutes"),
]

NORMAL_QUERIES = [
    ("vlog", "English daily vlog 5 minutes"),
    ("vlog", "English travel vlog 8 minutes"),
    ("vlog", "English study vlog 6 minutes"),
    ("vlog", "English cooking vlog 7 minutes"),
    ("documentary", "English mini documentary 8 minutes"),
    ("documentary", "BBC documentary short English 7 minutes"),
    ("documentary", "history documentary English 8 minutes"),
    ("documentary", "nature documentary English 8 minutes"),
    ("podcast", "English podcast conversation 8 minutes"),
    ("podcast", "science podcast English 7 minutes"),
    ("podcast", "news podcast English 8 minutes"),
    ("interview", "English interview 8 minutes"),
    ("technology", "technology explanation English 8 minutes"),
    ("education", "educational explanation English 7 minutes"),
    ("review", "product review English 8 minutes"),
    ("news", "English news explainer 6 minutes"),
    ("comedy", "English comedy sketch 6 minutes"),
    ("sports", "sports analysis English 8 minutes"),
    ("music", "English music lesson 7 minutes"),
    ("DIY", "DIY tutorial English 8 minutes"),
]


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def kids_genre(query: str) -> str:
    q = query.lower()
    for terms, label in [
        (("song", "rhymes", "music", "bum", "goose"), "music"),
        (("story", "read aloud", "peppa"), "stories/shows"),
        (("science", "experiment", "scishow", "nat geo", "national geographic", "brainpop", "minuteearth"), "science/learning"),
        (("draw", "craft"), "art/crafts"),
        (("yoga",), "movement/yoga"),
        (("numberblocks", "math"), "math"),
    ]:
        if any(term in q for term in terms):
            return label
    return "educational/entertainment"


def search_set(ydl: YoutubeDL, queries: list[tuple[str, str]], target: int, source: str) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    per_query = max(12, (target // len(queries)) + 8)
    for group, query in queries:
        try:
            result = ydl.extract_info(f"ytsearch{per_query}:{query}", download=False)
        except Exception as exc:
            print(f"WARN search failed ({query!r}): {exc}")
            continue
        for item in (result or {}).get("entries", []):
            if not item:
                continue
            video_id = clean(item.get("id"))
            duration = item.get("duration")
            if not video_id or video_id in seen or not isinstance(duration, (int, float)):
                continue
            if not MIN_SECONDS <= duration <= MAX_SECONDS:
                continue
            title = clean(item.get("title"))
            url = item.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
            seen.add(video_id)
            rows.append({
                "source": source,
                "classification": "kids_candidate" if source == "youtube_kids" else "normal_youtube",
                "age_group": group if source == "youtube_kids" else "adult/general",
                "genre": kids_genre(query) if source == "youtube_kids" else group,
                "title": title,
                "channel": clean(item.get("channel") or item.get("uploader")),
                "duration_seconds": int(duration),
                "duration_min": round(duration / 60, 2),
                "video_id": video_id,
                "url": url,
                "kids_url": f"https://www.youtubekids.com/watch/{video_id}" if source == "youtube_kids" else "",
                "search_query": query,
            })
            if len(rows) >= target:
                return rows
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with YoutubeDL(opts) as ydl:
        kids = search_set(ydl, KIDS_QUERIES, args.target, "youtube_kids")
        normal = search_set(ydl, NORMAL_QUERIES, args.target, "youtube")
    rows = kids + normal
    manifest = args.out_dir / "youtube_manifest.csv"
    fields = list(rows[0]) if rows else ["source", "url"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "youtube_manifest.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, subset in (("videos_kids.txt", kids), ("videos_normal.txt", normal), ("videos.txt", rows)):
        (args.out_dir / name).write_text("# Metadata collected by collect_youtube_manifest.py\n" + "\n".join(r["url"] for r in subset) + "\n", encoding="utf-8")
    print(f"YouTube Kids candidates: {len(kids)}/{args.target}")
    print(f"Normal YouTube videos:   {len(normal)}/{args.target}")
    print(f"Manifest: {manifest.resolve()}")
    return 0 if len(kids) == args.target and len(normal) == args.target else 2


if __name__ == "__main__":
    raise SystemExit(main())
