"""
Smoke test for speech_pipeline.py
=================================

CPU-friendly, staged validation. Each stage prints PASS / FAIL / SKIP.

  * Stages 1-3 are lightweight (numpy + parselmouth only) and validate the
    thesis-critical acoustic + syllable math against a KNOWN synthetic signal.
  * Stages 4-5 (Demucs, WhisperX) are heavy; they SKIP unless deps are present
    and you pass --heavy. On a CPU laptop they are slow — use a short clip.

Usage
-----
    python smoke_test.py                        # light stages only
    python smoke_test.py --heavy                # + Demucs/WhisperX if installed
    python smoke_test.py --heavy --audio my.wav # transcribe a real short clip
    python smoke_test.py --heavy --whisper-model tiny   # fastest on CPU

A synthetic test tone is generated automatically; --audio overrides it for ASR.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import struct
import sys
import wave
from pathlib import Path

# --- tiny test harness ---------------------------------------------------- #
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_results: list[tuple[str, str, str]] = []


def record(stage: str, status: str, detail: str = "") -> None:
    color = {"PASS": "\033[92m", "FAIL": "\033[91m", "SKIP": "\033[93m"}.get(status, "")
    reset = "\033[0m" if color else ""
    print(f"  [{color}{status:4}{reset}] {stage}" + (f" — {detail}" if detail else ""))
    _results.append((stage, status, detail))


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


# --- synthetic signal ----------------------------------------------------- #
def make_test_tone(path: Path, sr: int = 44_100, seconds: float = 1.0,
                   f0: float = 220.0, vibrato_hz: float = 5.0,
                   vibrato_depth: float = 8.0) -> None:
    """
    Write a mono WAV of a 220 Hz tone with 5 Hz vibrato (+/-8 Hz).
    Ground truth: pitch mean ~= 220 Hz, pitch variance clearly > 0.
    Pure stdlib (math + wave) so it needs no numpy.
    """
    import math

    n = int(sr * seconds)
    phase = 0.0
    frames = bytearray()
    for i in range(n):
        t = i / sr
        inst_f = f0 + vibrato_depth * math.sin(2 * math.pi * vibrato_hz * t)
        phase += 2 * math.pi * inst_f / sr
        sample = int(0.6 * 32767 * math.sin(phase))
        frames += struct.pack("<h", max(-32768, min(32767, sample)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(bytes(frames))


# --- stages --------------------------------------------------------------- #
def stage_import() -> bool:
    """Stage 0: module imports without the GPU stack (proves lazy imports work)."""
    try:
        import speech_pipeline  # noqa: F401
        record("import speech_pipeline (no torch)", PASS)
        return True
    except Exception as exc:
        record("import speech_pipeline", FAIL, repr(exc))
        return False


def stage_syllables() -> None:
    """Stage 1: syllable heuristic against known counts and splits."""
    from speech_pipeline import AcousticAnalyzer as A

    cases = {
        "cat": ["cat"],
        "hello": ["hel", "lo"],
        "spectacular": ["spec", "ta", "cu", "lar"],
        "beautiful": ["beau", "ti", "ful"],
        "make": ["make"],
        "e": ["e"],
        "": [""],
    }
    bad = {word: A.syllabify(word) for word, expected in cases.items()
           if A.syllabify(word) != expected}
    if bad:
        record("syllabify/count_syllables", FAIL, f"mismatches: {bad}")
    else:
        record("syllabify/count_syllables", PASS, f"{len(cases)} words correct")


def stage_acoustic(tone: Path) -> None:
    """Stage 2: F0 on a KNOWN 220 Hz tone -> mean ~220, variance > 0."""
    if not have("parselmouth"):
        record("AcousticAnalyzer pitch", SKIP, "parselmouth not installed")
        return
    from speech_pipeline import AcousticAnalyzer

    an = AcousticAnalyzer(tone, pitch_floor=75, pitch_ceiling=600)
    # one "word" spanning the whole tone
    voiced = an._pitch_in_window(0.05, 0.95)
    if voiced.size == 0:
        record("AcousticAnalyzer pitch", FAIL, "no voiced frames detected")
        return
    mean = float(voiced.mean())
    var = float(voiced.var())
    ok = 200 <= mean <= 240 and var > 0.5
    record("AcousticAnalyzer pitch", PASS if ok else FAIL,
           f"mean={mean:.1f}Hz (exp~220), var={var:.1f} (exp>0)")


def stage_enrich(tone: Path) -> None:
    """Stage 3: full per-word enrichment -> schema + pause/rate arithmetic."""
    if not have("parselmouth"):
        record("AcousticAnalyzer.enrich", SKIP, "parselmouth not installed")
        return
    from speech_pipeline import AcousticAnalyzer

    words = [
        {"word": "hello", "start": 0.10, "end": 0.40},
        {"word": "spectacular", "start": 0.60, "end": 0.95},  # 0.20 s pause before
    ]
    feats = AcousticAnalyzer(tone).enrich(words)

    required = {"word", "start_time", "end_time", "pitch_mean_hz",
                "pitch_variance", "pause_after_sec", "speech_rate_sps"}
    d0, d1 = feats[0].to_dict(), feats[1].to_dict()

    problems = []
    if set(d0) != required:
        problems.append(f"schema mismatch: {set(d0) ^ required}")
    # pause_after of word 0 = 0.60 - 0.40 = 0.20
    if abs((d0["pause_after_sec"] or -1) - 0.20) > 1e-3:
        problems.append(f"pause_after={d0['pause_after_sec']} (exp 0.20)")
    # last word pause must be null
    if d1["pause_after_sec"] is not None:
        problems.append(f"last pause={d1['pause_after_sec']} (exp None)")
    # speech rate word0: 2 syllables / 0.30 s ~= 6.67
    if abs((d0["speech_rate_sps"] or 0) - 6.667) > 0.05:
        problems.append(f"rate={d0['speech_rate_sps']} (exp ~6.67)")

    record("AcousticAnalyzer.enrich", FAIL if problems else PASS,
           "; ".join(problems) or "schema + pause + rate correct")


def stage_syllable_enrich(tone: Path) -> None:
    """Stage 3b: syllable windows cover their parent words without gaps."""
    if not have("parselmouth"):
        record("AcousticAnalyzer.enrich_syllables", SKIP, "parselmouth not installed")
        return
    from speech_pipeline import AcousticAnalyzer

    words = [
        {"word": "hello", "start": 0.10, "end": 0.40},
        {"word": "cat", "start": 0.60, "end": 0.90},
    ]
    feats = AcousticAnalyzer(tone).enrich_syllables(words)
    problems = []
    if [f.syllable for f in feats] != ["hel", "lo", "cat"]:
        problems.append(f"unexpected units: {[f.syllable for f in feats]}")
    if abs(feats[0].start_time - 0.10) > 1e-3 or abs(feats[1].end_time - 0.40) > 1e-3:
        problems.append("syllable windows do not cover parent word")
    if feats[0].pause_after_sec != 0.0 or feats[1].pause_after_sec != 0.20:
        problems.append("internal/inter-word pauses are incorrect")
    record("AcousticAnalyzer.enrich_syllables", FAIL if problems else PASS,
           "; ".join(problems) or "segmentation + timing + pause correct")


def stage_demucs(tone: Path, workdir: Path, run_heavy: bool) -> None:
    if not run_heavy:
        record("VocalIsolator (Demucs)", SKIP, "use --heavy to run (slow on CPU)")
        return
    if not (have("demucs") and have("torch") and have("torchaudio")):
        record("VocalIsolator (Demucs)", SKIP, "demucs/torch not installed")
        return
    from speech_pipeline import VocalIsolator

    try:
        out = VocalIsolator("cpu").isolate(tone, workdir / "vocals_test.wav")
        ok = out.exists() and out.stat().st_size > 0
        record("VocalIsolator (Demucs)", PASS if ok else FAIL,
               f"wrote {out.name}" if ok else "no output")
    except Exception as exc:
        record("VocalIsolator (Demucs)", FAIL, repr(exc))


def stage_whisperx(audio: Path, run_heavy: bool, model: str) -> None:
    if not run_heavy:
        record("GPUTranscriber (WhisperX)", SKIP, "use --heavy to run (slow on CPU)")
        return
    if not have("whisperx"):
        record("GPUTranscriber (WhisperX)", SKIP, "whisperx not installed")
        return
    from speech_pipeline import GPUTranscriber

    try:
        # int8 is the CPU-friendly compute type; tiny model = fastest.
        tr = GPUTranscriber("cpu", model_size=model, compute_type="int8")
        words = tr.transcribe(audio)
        # A synthetic tone yields no words; a real clip should yield some.
        record("GPUTranscriber (WhisperX)", PASS,
               f"ran; {len(words)} words (0 is expected for a pure tone)")
    except Exception as exc:
        record("GPUTranscriber (WhisperX)", FAIL, repr(exc))


# --- main ----------------------------------------------------------------- #
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--heavy", action="store_true", help="Also run Demucs/WhisperX.")
    p.add_argument("--audio", type=Path, default=None,
                   help="Real short WAV for the ASR stage (defaults to synthetic tone).")
    p.add_argument("--whisper-model", default="tiny", help="Fastest on CPU: tiny/base.")
    args = p.parse_args()

    workdir = Path("./_smoke_run")
    workdir.mkdir(exist_ok=True)
    tone = workdir / "tone_220hz.wav"

    print("\n=== Environment ===")
    for m in ["numpy", "parselmouth", "torch", "torchaudio", "demucs", "whisperx"]:
        record(m, PASS if have(m) else SKIP, "installed" if have(m) else "not installed")
    for b in ["ffmpeg", "yt-dlp"]:
        record(b, PASS if shutil.which(b) else SKIP,
               "on PATH" if shutil.which(b) else "not on PATH")

    print("\n=== Generating synthetic 220 Hz test tone ===")
    make_test_tone(tone)
    record("make_test_tone", PASS if tone.exists() else FAIL, str(tone))

    print("\n=== Light stages (CPU, no GPU stack) ===")
    if not stage_import():
        print("\nCannot import pipeline; aborting.")
        return 1
    stage_syllables()
    stage_acoustic(tone)
    stage_enrich(tone)
    stage_syllable_enrich(tone)

    print("\n=== Heavy stages ===")
    stage_demucs(tone, workdir, args.heavy)
    stage_whisperx(args.audio or tone, args.heavy, args.whisper_model)

    # summary
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    n_skip = sum(1 for _, s, _ in _results if s == SKIP)
    print(f"\n=== Summary: {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP ===")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
