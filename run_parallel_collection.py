"""Parallel discovery + no-download verification wrapper for a fast manifest."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse, random
from yt_dlp import YoutubeDL
from collect_fast_verified_manifest import KIDS, NORMAL, MIN_SECONDS, MAX_SECONDS, YEARS, clean, verify, write

def search_one(task):
    source, genre, phrase, year = task
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}) as ydl:
            result = ydl.extract_info(f"ytsearch12:{phrase} {year}", download=False)
    except Exception:
        return []
    out = []
    for item in (result or {}).get("entries", []):
        vid, duration = clean(item.get("id")), item.get("duration")
        if vid and isinstance(duration, (int, float)) and MIN_SECONDS <= duration <= MAX_SECONDS:
            out.append({"source": source, "classification": "kids_candidate" if source == "youtube_kids" else "normal_youtube", "genre": genre, "video_id": vid, "url": item.get("webpage_url") or f"https://www.youtube.com/watch?v={vid}", "title": clean(item.get("title")), "channel": clean(item.get("channel") or item.get("uploader")), "duration_seconds": int(duration), "search_year": year, "search_query": f"{phrase} {year}"})
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--target",type=int,default=1000);p.add_argument("--out-dir",type=Path,default=Path("."));p.add_argument("--search-workers",type=int,default=12);p.add_argument("--verify-workers",type=int,default=20);a=p.parse_args(); a.out_dir.mkdir(exist_ok=True)
    tasks=[("youtube_kids",g,q,y) for y in YEARS for g,q in KIDS]+[("youtube",g,q,y) for y in YEARS for g,q in NORMAL]
    random.Random(20260920).shuffle(tasks); candidates=[]; seen=set()
    with ThreadPoolExecutor(max_workers=a.search_workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(search_one,t) for t in tasks]),1):
            for row in f.result():
                if row["video_id"] not in seen: seen.add(row["video_id"]);candidates.append(row)
            if i%50==0: print(f"searches={i}/{len(tasks)} candidates={len(candidates)}",flush=True)
    random.Random(20260920).shuffle(candidates); kids=[];normal=[];done=0
    pool=ThreadPoolExecutor(max_workers=a.verify_workers); futures={pool.submit(verify,row):row["source"] for row in candidates}
    for f in as_completed(futures):
        done+=1; row=f.result(); source=futures[f]
        if row and source=="youtube_kids" and len(kids)<a.target: kids.append(row)
        elif row and source=="youtube" and len(normal)<a.target: normal.append(row)
        if done%25==0:
            write(a.out_dir,kids,normal,done,len(candidates));print(f"checked={done} kids={len(kids)}/{a.target} normal={len(normal)}/{a.target}",flush=True)
        if len(kids)>=a.target and len(normal)>=a.target:
            for x in futures: x.cancel()
            pool.shutdown(wait=False,cancel_futures=True);break
    else: pool.shutdown(wait=True)
    write(a.out_dir,kids,normal,done,len(candidates));print(f"final kids={len(kids)} normal={len(normal)}",flush=True)
    return 0 if len(kids)==a.target and len(normal)==a.target else 2
if __name__=="__main__": raise SystemExit(main())
