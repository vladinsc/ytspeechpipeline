"""Render a karaoke-style review video from a pipeline result JSON.

The video shows the original audio over a plain background with three tracks:

* the current line of words, each highlighted at its aligned start time;
* the syllables of that line, highlighted at their (estimated) times;
* the measured prosodic features of the word being spoken.

Usage
-----
    python tools/render_karaoke_video.py results/kids/video_0069_2njn71TqkjA.json \
        "data/youtube kids/What Earth in 2050 could look like - Shannon Odell [2njn71TqkjA].mp3" \
        --out karaoke.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

WIDTH, HEIGHT = 1280, 720
MAX_LINE_CHARS = 38
LINE_BREAK_PAUSE = 0.6  # seconds of silence that force a new line
LEAD_IN = 0.25          # show a line slightly before its first word
LINGER = 0.4            # keep a line slightly after its last word

# ASS colours are &HAABBGGRR
YELLOW = "&H0045D4FF"
CYAN = "&H00FFC85C"
GREY = "&H00A6948A"
DIM = "&H0072635A"
LIGHT = "&H00E6DED8"


def ass_time(t: float) -> str:
    t = max(t, 0.0)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")")


def karaoke(segments: list[tuple[float, float, str]], line_start: float) -> str:
    """Build a \\k karaoke string; gaps between segments become empty segments."""
    out, cursor = [], int(round(line_start * 100))
    for start, end, text in segments:
        s, e = int(round(start * 100)), int(round(end * 100))
        if s > cursor:
            out.append(f"{{\\k{s - cursor}}}")
            cursor = s
        e = max(e, cursor)
        out.append(f"{{\\k{e - cursor}}}{ass_escape(text)}")
        cursor = e
    return "".join(out)


def group_lines(words: list[dict]) -> list[list[int]]:
    lines, current, chars = [], [], 0
    for i, w in enumerate(words):
        if current:
            prev = words[current[-1]]
            gap = w["start_time"] - prev["end_time"]
            sentence_end = prev["word"].rstrip('"\'').endswith((".", "?", "!"))
            if gap > LINE_BREAK_PAUSE or sentence_end or chars + len(w["word"]) + 1 > MAX_LINE_CHARS:
                lines.append(current)
                current, chars = [], 0
        current.append(i)
        chars += len(w["word"]) + 1
    if current:
        lines.append(current)
    return lines


def fmt(value, spec: str, unit: str) -> str:
    return "—" if value is None else f"{value:{spec}} {unit}"


def build_ass(result: dict, title: str) -> str:
    words = sorted(result["words"], key=lambda w: w["start_time"])
    syllables = sorted(result.get("syllables", []), key=lambda s: s["start_time"])
    lines = group_lines(words)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {WIDTH}
PlayResY: {HEIGHT}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Title,Segoe UI,28,{GREY},{GREY},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,40,40,40,1
Style: Label,Segoe UI,20,{DIM},{DIM},&H00000000,&H00000000,1,0,0,0,100,100,2,0,1,0,0,8,40,40,0,1
Style: Words,Segoe UI,56,{YELLOW},{GREY},&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,0,0,8,40,40,250,1
Style: Syll,Consolas,34,{CYAN},{DIM},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,40,40,390,1
Style: Info,Segoe UI,26,{LIGHT},{LIGHT},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,8,40,40,560,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    end_of_audio = words[-1]["end_time"] + 5 if words else 5
    events = [
        f"Dialogue: 0,{ass_time(0)},{ass_time(end_of_audio)},Title,,0,0,0,,{ass_escape(title)}",
        f"Dialogue: 0,{ass_time(0)},{ass_time(end_of_audio)},Label,,0,0,215,,WORDS · WhisperX forced alignment",
        f"Dialogue: 0,{ass_time(0)},{ass_time(end_of_audio)},Label,,0,0,355,,SYLLABLES · estimated within each word",
        f"Dialogue: 0,{ass_time(0)},{ass_time(end_of_audio)},Label,,0,0,525,,PROSODY · Praat / Parselmouth",
    ]

    prev_end = 0.0
    for n, idx in enumerate(lines):
        first, last = words[idx[0]], words[idx[-1]]
        start = max(first["start_time"] - LEAD_IN, prev_end)
        next_start = words[lines[n + 1][0]]["start_time"] - LEAD_IN if n + 1 < len(lines) else last["end_time"] + 2
        end = max(min(last["end_time"] + LINGER, next_start), last["end_time"])
        prev_end = end

        word_segments = [(words[i]["start_time"], words[i]["end_time"],
                          words[i]["word"] + (" " if i != idx[-1] else "")) for i in idx]
        events.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Words,,0,0,0,,{karaoke(word_segments, start)}")

        line_sylls = [s for s in syllables
                      if first["start_time"] - 0.005 <= s["start_time"] <= last["end_time"] + 0.005]
        if line_sylls:
            syl_segments = []
            for s in line_sylls:
                last_in_word = s["syllable_index"] == s["syllable_count"]
                syl_segments.append((s["start_time"], s["end_time"],
                                     s["syllable"] + ("  " if last_in_word else "·")))
            events.append(f"Dialogue: 1,{ass_time(start)},{ass_time(end)},Syll,,0,0,0,,{karaoke(syl_segments, start)}")

    for i, w in enumerate(words):
        nxt = words[i + 1]["start_time"] if i + 1 < len(words) else w["end_time"] + 1
        w_end = max(min(w["end_time"] + 0.3, nxt), w["end_time"])
        info = (f"“{ass_escape(w['word'])}”    pitch {fmt(w.get('pitch_mean_hz'), '.0f', 'Hz')}"
                f"    pause after {fmt(w.get('pause_after_sec'), '.2f', 's')}"
                f"    rate {fmt(w.get('speech_rate_sps'), '.1f', 'syl/s')}")
        events.append(f"Dialogue: 2,{ass_time(w['start_time'])},{ass_time(w_end)},Info,,0,0,0,,{info}")

    return header + "\n".join(events) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("result", type=Path, help="Pipeline result JSON (granularity word or both)")
    p.add_argument("audio", type=Path, help="Audio or video file of the same YouTube video")
    p.add_argument("--out", type=Path, default=Path("karaoke.mp4"))
    p.add_argument("--title", default=None)
    args = p.parse_args()

    result = json.loads(args.result.read_text(encoding="utf-8"))
    title = args.title or f"{result.get('label', '')} · {result.get('source_url', '')}"
    out = args.out.resolve()
    ass_path = out.with_suffix(".ass")
    ass_path.write_text(build_ass(result, title), encoding="utf-8")

    # Run ffmpeg from the subtitle's folder so the filter argument needs no path escaping.
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "lavfi", "-i", f"color=c=0x181410:s={WIDTH}x{HEIGHT}:r=30",
        "-i", str(args.audio.resolve()),
        "-vf", f"ass={ass_path.name}",
        "-map", "0:v", "-map", "1:a", "-shortest",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True, cwd=ass_path.parent)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
