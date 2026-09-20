"""Fast, checkpointed collection of public ~five-minute YouTube candidates.

No media is downloaded: yt-dlp only resolves metadata and formats. A retained
row has an audio-bearing format available without cookies, browser login, or
YouTube OAuth. Results are checkpointed after every 25 verified URLs.
"""
from __future__ import annotations

import argparse, csv, json, random, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from yt_dlp import YoutubeDL

MIN_SECONDS, MAX_SECONDS = 270, 330  # 4:30 to 5:30
YEARS = list(range(2008, datetime.now(UTC).year + 1))
KIDS = [
    ("cartoon", "kids cartoon English full episode"), ("music", "kids songs nursery rhymes English"),
    ("story", "kids read aloud story English"), ("science", "kids science video English"),
    ("nature", "kids animals nature English"), ("math", "kids math lesson English"),
    ("art", "kids drawing tutorial English"), ("craft", "kids crafts English"),
    ("movement", "kids yoga dance English"), ("education", "educational video for kids English"),
]
NORMAL = [
    ("vlog", "English daily vlog"), ("travel_vlog", "English travel vlog"),
    ("study_vlog", "English study vlog"), ("documentary", "short documentary English"),
    ("history_documentary", "short history documentary English"), ("nature_documentary", "short nature documentary English"),
    ("science_documentary", "short science documentary English"), ("interview", "English interview"),
    ("podcast", "English podcast conversation"), ("technology", "technology explainer English"),
    ("education", "educational explainer English"), ("news", "news explainer English"),
    ("comedy", "English comedy sketch"), ("sports", "sports analysis English"),
    ("diy", "DIY tutorial English"), ("gaming", "English gaming commentary"),
]

def clean(v: object) -> str:
    return re.sub(r"\s+", " ", str(v or "")).strip()

def discover(topics, target, source):
    """Search topic/year combinations, then randomize candidates."""
    rows, seen = [], set()
    requests = [(genre, phrase, year) for year in YEARS for genre, phrase in topics]
    random.Random(20260920).shuffle(requests)
    need = target * 3
    with YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}) as ydl:
        for genre, phrase, year in requests:
            try:
                result = ydl.extract_info(f"ytsearch12:{phrase} {year}", download=False)
            except Exception:
                continue
            for item in (result or {}).get("entries", []):
                vid, duration = clean(item.get("id")), item.get("duration")
                if not vid or vid in seen or not isinstance(duration, (int, float)) or not MIN_SECONDS <= duration <= MAX_SECONDS:
                    continue
                seen.add(vid)
                rows.append({"source": source, "classification": "kids_candidate" if source == "youtube_kids" else "normal_youtube", "genre": genre, "video_id": vid, "url": item.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}", "title": clean(item.get("title")), "channel": clean(item.get("channel") or item.get("uploader")), "duration_seconds": int(duration), "search_year": year, "search_query": f"{phrase} {year}"})
                if len(rows) >= need:
                    random.Random(vid).shuffle(rows)
                    return rows
    random.Random(20260920).shuffle(rows)
    return rows

def verify(row):
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(row["url"], download=False)
    except Exception:
        return None
    duration, language = info.get("duration"), clean(info.get("language")).lower() if info else (None, "")
    if not info or not isinstance(duration, (int, float)) or not MIN_SECONDS <= duration <= MAX_SECONDS:
        return None
    if language and not language.startswith("en"):
        return None
    if info.get("availability") in {"private", "premium_only", "subscriber_only"}:
        return None
    if not any(f.get("url") and f.get("acodec") not in (None, "none") for f in info.get("formats", [])):
        return None
    return {**row, "title": clean(info.get("title")) or row["title"], "channel": clean(info.get("channel") or info.get("uploader")) or row["channel"], "duration_seconds": int(duration), "upload_date": info.get("upload_date") or "", "language": language or "english_search_candidate", "download_check": "passed_unauthenticated_metadata_format_check", "download_checked_at": datetime.now(UTC).isoformat(timespec="seconds")}

def write(out, kids, normal, done, total):
    rows = kids + normal
    fields = list(rows[0]) if rows else ["url"]
    with (out / "youtube_manifest.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (out / "youtube_manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    header = "# Metadata-only unauthenticated yt-dlp format check; media not downloaded\n"
    (out / "videos_kids.txt").write_text(header + "\n".join(x["url"] for x in kids) + "\n", encoding="utf-8")
    (out / "videos_normal.txt").write_text(header + "\n".join(x["url"] for x in normal) + "\n", encoding="utf-8")
    (out / "videos.txt").write_text(header + "# LABEL: YOUTUBE_KIDS\n" + "\n".join(x["url"] for x in kids) + "\n# LABEL: NORMAL_YOUTUBE\n" + "\n".join(x["url"] for x in normal) + "\n", encoding="utf-8")
    (out / "collection_progress.json").write_text(json.dumps({"checked": done, "candidate_count": total, "kids_verified": len(kids), "normal_verified": len(normal), "updated_at": datetime.now(UTC).isoformat()}, indent=2), encoding="utf-8")

def main():
    p = argparse.ArgumentParser(); p.add_argument("--target", type=int, default=1000); p.add_argument("--workers", type=int, default=10); p.add_argument("--out-dir", type=Path, default=Path(".")); a = p.parse_args(); a.out_dir.mkdir(exist_ok=True)
    kids_c, normal_c = discover(KIDS, a.target, "youtube_kids"), discover(NORMAL, a.target, "youtube")
    tagged = [("kids", r) for r in kids_c] + [("normal", r) for r in normal_c]
    random.Random(20260920).shuffle(tagged)
    kids, normal, done = [], [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(verify, row): label for label, row in tagged}
        for future in as_completed(futures):
            done += 1; label, row = futures[future], future.result()
            if row and (label == "kids" and len(kids) < a.target or label == "normal" and len(normal) < a.target):
                (kids if label == "kids" else normal).append(row)
            if done % 25 == 0:
                write(a.out_dir, kids, normal, done, len(tagged)); print(f"checked={done} kids={len(kids)}/{a.target} normal={len(normal)}/{a.target}", flush=True)
            if len(kids) >= a.target and len(normal) >= a.target:
                break
    write(a.out_dir, kids, normal, done, len(tagged)); print(f"final kids={len(kids)} normal={len(normal)}", flush=True)
    return 0 if len(kids) == a.target and len(normal) == a.target else 2

if __name__ == "__main__": raise SystemExit(main())
