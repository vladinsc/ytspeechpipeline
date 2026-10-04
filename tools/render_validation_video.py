"""Render an annotated, time-synced validation video from feature JSONL.

The video uses the source recording's original audio, trimmed to at most ten
minutes. Text, words, phones, speakers, metrics, and quality flags come only
from recorded feature rows; unavailable fields are labelled as unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path


WIDTH = 1280
HEIGHT = 720
FPS = 15
MAX_SECONDS = 600.0


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _ass_time(seconds: float) -> str:
    centiseconds = max(0, round(seconds * 100))
    hours, rem = divmod(centiseconds, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cents = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cents:02d}"


def _text(value: object, limit: int = 240) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = value.replace("{", "(").replace("}", ")").replace("\\", "/")
    return value[:limit] + ("…" if len(value) > limit else "")


def _metric(value: object, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def _event(start: float, end: float, style: str, text: str, duration: float) -> str | None:
    start = max(0.0, start)
    end = min(duration, end)
    if end - start < 0.01:
        return None
    return f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,{_text(text)}"


def _record_source_id(record: dict) -> str | None:
    """Use the common batch ID, or derive a stable ID for Newman/Ratner rows."""
    if record.get("source_id"):
        return str(record["source_id"])
    if all(record.get(key) for key in ("recording_id", "source_section")):
        return "NewmanRatner:{age}:{recording}:{section}".format(
            age=record.get("age_group", "unknown"),
            recording=record["recording_id"], section=record["source_section"],
        )
    return None


def _audio_path(record: dict) -> str | None:
    return record.get("audio_path") or record.get("source_audio") or record.get("source")


def _ass_filter_path(path: Path) -> str:
    """Escape a local filename for FFmpeg's libass filter on Windows or Linux."""
    value = path.resolve().as_posix()
    return value.replace("\\", "/").replace(":", r"\:").replace("'", r"\'")


def _ass_header() -> str:
    styles = [
        ("Header", 26, "&H00FFFFFF", 24, 7),
        ("Speaker", 23, "&H0077D5FF", 78, 7),
        ("Transcript", 26, "&H00FFFFFF", 150, 7),
        ("ASR", 22, "&H009EE7AA", 205, 7),
        ("Word", 31, "&H0000D7FF", 255, 7),
        ("Phone", 26, "&H00A8FFB0", 325, 7),
        ("Syllable", 23, "&H00C4B5FD", 365, 7),
        ("Features", 22, "&H00FFFFFF", 410, 7),
        ("Features2", 22, "&H00FFFFFF", 450, 7),
        ("Flags", 18, "&H0090A0B0", 505, 7),
    ]
    style_lines = [
        f"Style: {name},DejaVu Sans,{size},{color},&H00000000,&H00000000,&H99000000,"
        f"-1,0,0,0,100,100,0,0,1,1,0,{align},50,50,{margin},1"
        for name, size, color, margin, align in styles
    ]
    return ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n"
            "WrapStyle: 0\nScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
            "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
            "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
            + "\n".join(style_lines)
            + "\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")


def build_ass(records: list[dict], source_id: str, clip_start: float,
              clip_duration: float) -> tuple[str, dict]:
    """Build timed overlays; times in feature JSONL are original-audio times."""
    if not records:
        raise ValueError(f"No feature rows for {source_id}")
    dataset = records[0].get("dataset") or "Newman/Ratner"
    lines = []
    add = lambda a, b, style, value: lines.append(
        _event(a - clip_start, b - clip_start, style, value, clip_duration)
    )
    add(clip_start, clip_start + clip_duration, "Header",
        f"{dataset} | {source_id} | original audio, max 10 min")
    shown = 0
    word_events = 0
    phone_events = 0
    syllable_events = 0
    ordered_records = sorted(records, key=lambda item: item["start_sec"])
    for record_index, record in enumerate(ordered_records):
        start = float(record["start_sec"])
        end = float(record["end_sec"])
        if end <= clip_start or start >= clip_start + clip_duration:
            continue
        next_start = (
            float(ordered_records[record_index + 1]["start_sec"])
            if record_index + 1 < len(ordered_records) else end
        )
        visual_end = min(end, next_start)
        timeline_overlap = next_start < end
        shown += 1
        speaker = record.get("speaker_id") or record.get("speaker") or "unresolved"
        provenance = record.get("speaker_attribution") or "unknown"
        speaker_text = f"Speaker: {speaker} | attribution: {provenance}"
        if record.get("diarization_speaker"):
            speaker_text += f" | audio cluster: {record['diarization_speaker']}"
            if record.get("diarization_mapped_speaker"):
                speaker_text += f" (maps to {record['diarization_mapped_speaker']})"
        add(start, visual_end, "Speaker", speaker_text)
        transcript = record.get("transcript")
        if transcript:
            add(start, visual_end, "Transcript", f"Transcript: {_text(transcript, 170)}")
        else:
            add(start, visual_end, "Transcript", "No lexical transcript supplied for this clip")
        if record.get("independent_asr_text"):
            add(start, visual_end, "ASR", f"Independent ASR: {_text(record['independent_asr_text'], 170)}")
        acoustic = record.get("acoustic_features") or {}
        feature_text = (
            f"F0 median {_metric(acoustic.get('pitch_median_hz'), ' Hz')} | "
            f"F0 p05–p95 {_metric(acoustic.get('pitch_p05_hz'))}–"
            f"{_metric(acoustic.get('pitch_p95_hz'), ' Hz')} | "
            f"intensity {_metric(acoustic.get('intensity_mean_db'), ' dB')}"
        )
        add(start, visual_end, "Features", feature_text)
        add(start, visual_end, "Features2",
            f"Words/s {_metric(acoustic.get('words_per_sec'))} | "
            f"articulation {_metric(acoustic.get('articulation_fraction'))} | "
            f"voiced {_metric(acoustic.get('voiced_frame_fraction'))}")
        flags = record.get("quality_flags") or (record.get("alignment_qc") or {}).get("quality_flags") or []
        if timeline_overlap:
            flags = [*flags, "next CHAT turn overlaps; overlay switches to later turn"]
        if flags:
            add(start, visual_end, "Flags", "QC: " + ", ".join(flags))
        words = record.get("words") or []
        for index, word in enumerate(words):
            first = max(0, index - 4)
            last = min(len(words), index + 5)
            context = " ".join(
                ("[" + str(words[k]["word"]) + "]") if k == index
                else str(words[k]["word"])
                for k in range(first, last)
            )
            add(float(word["start_time"]), min(float(word["end_time"]), visual_end), "Word",
                f"Aligned word: {context} | F0 {_metric(word.get('pitch_mean_hz'), ' Hz')}"
                f" | F0 variance {_metric(word.get('pitch_variance'))}"
                f" | pause {_metric(word.get('pause_after_sec'), ' s')}"
                f" | rate {_metric(word.get('speech_rate_sps'), ' syll/s')}")
            word_events += 1
        for phone in record.get("phones") or []:
            add(float(phone["start"]), min(float(phone["end"]), visual_end), "Phone",
                f"Aligned phone: {phone['phone']} | word index {phone.get('word_index')}")
            phone_events += 1
        for syllable in record.get("syllables") or []:
            add(float(syllable["start_time"]), min(float(syllable["end_time"]), visual_end), "Syllable",
                f"Syllable: {syllable['syllable']} | parent {syllable['parent_word']}"
                f" | F0 {_metric(syllable.get('pitch_mean_hz'), ' Hz')}"
                f" | timing {syllable.get('timing_method', 'unknown')}")
            syllable_events += 1
        if not words:
            add(start, visual_end, "Word", "Word alignment unavailable")
        if not record.get("phones"):
            add(start, visual_end, "Phone", "Phone alignment unavailable")
        if not record.get("syllables"):
            add(start, visual_end, "Syllable", "Syllable timing unavailable")
    events = [line for line in lines if line]
    if shown == 0:
        raise ValueError(f"No feature rows overlap the video trim for {source_id}")
    return _ass_header() + "\n".join(events) + "\n", {
        "shown_utterances_or_clips": shown,
        "word_events": word_events,
        "phone_events": phone_events,
        "syllable_events": syllable_events,
        "subtitle_events": len(events),
    }


def render(source_id: str, records: list[dict], output: Path,
           max_seconds: float = MAX_SECONDS, clip_start: float = 0.0) -> dict:
    if not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("Validation videos must be at most 600 seconds")
    if clip_start < 0 or not math.isfinite(clip_start):
        raise ValueError("Video clip start must be a finite nonnegative time")
    audio_paths = {_audio_path(row) for row in records}
    audio_paths.discard(None)
    if len(audio_paths) != 1:
        raise ValueError(f"Expected one original audio file for {source_id}")
    audio = Path(next(iter(audio_paths)))
    if not audio.is_file():
        raise FileNotFoundError(audio)
    original_duration = _probe_duration(audio)
    duration = min(original_duration - clip_start, max_seconds)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid audio duration for {audio}")
    subtitles, counts = build_ass(records, source_id, clip_start, duration)
    output.parent.mkdir(parents=True, exist_ok=True)
    ass_path = output.with_suffix(".ass")
    ass_path.write_text(subtitles, encoding="utf-8-sig")
    filter_graph = (
        "[1:a]showwaves=s=1160x120:mode=line:r=15:colors=0x38bdf8[wave];"
        f"[0:v][wave]overlay=60:560:shortest=1,ass=filename='{_ass_filter_path(ass_path)}'[video]"
    )
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i",
        f"color=c=0x0d1524:s={WIDTH}x{HEIGHT}:r={FPS}:d={duration:.3f}",
        "-ss", f"{clip_start:.3f}", "-t", f"{duration:.3f}", "-i", str(audio),
        "-filter_complex", filter_graph,
        "-map", "[video]", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
        "-t", f"{duration:.3f}", "-shortest", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"Video render failed for {source_id}: {result.stderr[-2000:]}")
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format=duration",
         "-of", "json", str(output)],
        capture_output=True, text=True, check=True,
    )
    video_info = json.loads(probed.stdout)
    stream_types = {stream["codec_type"] for stream in video_info.get("streams", [])}
    output_duration = float(video_info["format"]["duration"])
    if (stream_types != {"audio", "video"} or output_duration > MAX_SECONDS + 0.5
            or output_duration < max(0.0, duration - 1.0)):
        raise RuntimeError(f"Rendered media failed duration/stream checks: {output}")
    info = {
        "source_id": source_id,
        "original_audio": str(audio),
        "original_duration_sec": round(original_duration, 3),
        "video_start_sec": clip_start,
        "video_duration_sec": round(duration, 3),
        "encoded_duration_sec": round(output_duration, 3),
        "video": str(output), "ass": str(ass_path),
        **counts,
    }
    output.with_suffix(".render.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    args = parser.parse_args()
    records = []
    with args.features.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if _record_source_id(row) == args.source_id:
                    records.append(row)
    print(json.dumps(render(args.source_id, records, args.output, args.max_seconds), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
