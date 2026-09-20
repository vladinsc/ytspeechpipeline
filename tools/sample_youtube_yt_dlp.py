#!/usr/bin/env python3
"""Sample YouTube / YouTube Kids-oriented videos and test unauthenticated yt-dlp access.

This intentionally uses yt-dlp's public search extractor and never supplies cookies,
login credentials, or browser profiles. It performs metadata/format extraction only
(`--simulate`), so it does not save media files.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

KIDS_QUERIES = [
    "nursery rhymes", "preschool learning", "kids songs", "cartoons for kids",
    "educational videos for children", "science for kids", "storytime for kids",
    "toy review kids", "family friendly animation", "phonics for children",
    "math for kids", "animals for kids", "drawing for kids", "kids crafts",
    "bedtime stories for kids", "music for children", "kids exercise", "lego kids",
]
GENERAL_QUERIES = [
    "music", "news", "gaming", "sports", "technology", "cooking", "travel",
    "documentary", "comedy", "education", "science", "podcast", "fitness",
    "DIY", "history", "reviews", "live", "short film", "nature", "cars",
]


def run_json(cmd: list[str], timeout: int = 45) -> tuple[dict | None, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    text = (p.stdout or "").strip()
    if not text:
        return None, (p.stderr or "").strip()[-1000:]
    try:
        return json.loads(text.splitlines()[-1]), (p.stderr or "").strip()[-1000:]
    except json.JSONDecodeError:
        return None, (p.stderr or text).strip()[-1000:]


def discover(kind: str, target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    queries = KIDS_QUERIES if kind == "kids" else GENERAL_QUERIES
    found: dict[str, dict] = {}
    # Oversample because search results can contain unavailable/private/duplicate items.
    while len(found) < target:
        q = rng.choice(queries)
        # Search result count is capped per request; changing query order/terms gives a
        # broad, reproducible pseudo-random sample without authenticated API access.
        n = min(50, max(10, target - len(found) + 10))
        cmd = ["yt-dlp", "--flat-playlist", "--dump-single-json", "--skip-download",
               f"ytsearch{n}:{q}"]
        data, err = run_json(cmd, timeout=90)
        if not data:
            raise RuntimeError(f"search failed for {q!r}: {err}")
        entries = data.get("entries") or []
        rng.shuffle(entries)
        for e in entries:
            vid = e.get("id")
            if not vid or vid in found:
                continue
            found[vid] = {
                "kind": kind,
                "video_id": vid,
                "url": f"https://www.youtube.com/watch?v={vid}",
                "search_query": q,
                "title_discovered": e.get("title"),
                "channel_discovered": e.get("channel") or e.get("uploader"),
            }
            if len(found) >= target:
                break
        if len(found) < target:
            time.sleep(0.5)
    return list(found.values())


def test_one(row: dict) -> dict:
    cmd = ["yt-dlp", "--simulate", "--skip-download", "--no-playlist",
           "--no-warnings", "--dump-single-json", row["url"]]
    started = time.time()
    try:
        data, err = run_json(cmd, timeout=60)
        row.update({
            "yt_dlp_ok": bool(data),
            "yt_dlp_error": "" if data else err,
            "title": (data or {}).get("title"),
            "duration": (data or {}).get("duration"),
            "availability": (data or {}).get("availability"),
            "tested_seconds": round(time.time() - started, 2),
        })
    except subprocess.TimeoutExpired:
        row.update({"yt_dlp_ok": False, "yt_dlp_error": "timeout", "tested_seconds": 60})
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--out", type=Path, default=Path("artifacts/youtube_yt_dlp_sample"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for kind in ("kids", "youtube"):
        print(f"Discovering {args.count} {kind} videos...", flush=True)
        rows = discover(kind, args.count, args.seed + (0 if kind == "kids" else 1))
        print(f"Testing {len(rows)} {kind} URLs with unauthenticated yt-dlp...", flush=True)
        done = 0
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(test_one, row) for row in rows]
            for future in as_completed(futures):
                all_rows.append(future.result())
                done += 1
                if done % 25 == 0 or done == len(rows):
                    print(f"  {kind}: {done}/{len(rows)}", flush=True)
    jsonl = args.out.with_suffix(".jsonl")
    csv_path = args.out.with_suffix(".csv")
    with jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = sorted({k for r in all_rows for k in r})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_rows)
    summary = {}
    for kind in ("kids", "youtube"):
        rows = [r for r in all_rows if r["kind"] == kind]
        summary[kind] = {"total": len(rows), "yt_dlp_ok": sum(bool(r.get("yt_dlp_ok")) for r in rows)}
    summary["seed"] = args.seed
    summary["note"] = "Kids rows are YouTube URLs discovered from kid-oriented queries; no authentication/cookies used."
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
