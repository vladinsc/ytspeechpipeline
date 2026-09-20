"""
CDS Prosody Extraction Pipeline
================================

A GPU-accelerated pipeline that turns a YouTube URL into a word-level transcript
enriched with prosodic / acoustic features, for comparing Child-Directed Speech
(YouTube Kids) against adult-directed speech.

Stages
------
1. AudioDownloader  : yt-dlp -> ffmpeg -> WAV
2. VocalIsolator    : Demucs (htdemucs) on CUDA  -> isolated vocals
3. GPUTranscriber   : WhisperX (large-v3) + explicit VAD + forced alignment
                    -> word timestamps
4. AcousticAnalyzer : Parselmouth (Praat) F0 + pause + speech-rate per word

The output granularity can be `word`, `syllable`, or `both`. Syllable boundaries
and timestamps are estimated within each aligned word with a documented English
orthographic heuristic; they are not phoneme-level forced alignments.

Design note on sample rates
---------------------------
`htdemucs` is trained on 44.1 kHz STEREO. Feeding it 16 kHz mono degrades
separation quality. This pipeline therefore:

  * decodes the download to 44.1 kHz stereo for Demucs,
  * keeps the isolated vocals at 44.1 kHz for pitch analysis (finer F0 grid),
  * produces a 16 kHz mono copy of the vocals ONLY for WhisperX.

All downstream analysis runs on the *isolated vocals*, never the raw audio.

Usage
-----
    python speech_pipeline.py "https://youtu.be/XXXX" --out result.json --workdir ./run
    python speech_pipeline.py "https://youtu.be/XXXX" --pitch-ceiling 800   # exaggerated CDS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

# NOTE: torch / torchaudio / demucs / whisperx / parselmouth are imported LAZILY
# inside the stages that use them, so lightweight stages (e.g. AcousticAnalyzer,
# syllable counting) can be imported and tested without the full GPU stack.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class WordFeature:
    word: str
    start_time: float
    end_time: float
    pitch_mean_hz: Optional[float] = None
    pitch_variance: Optional[float] = None
    pause_after_sec: Optional[float] = None
    speech_rate_sps: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SyllableFeature:
    syllable: str
    parent_word: str
    syllable_index: int
    syllable_count: int
    start_time: float
    end_time: float
    pitch_mean_hz: Optional[float] = None
    pitch_variance: Optional[float] = None
    pause_after_sec: Optional[float] = None
    speech_rate_sps: Optional[float] = None
    timing_method: str = "uniform_within_word"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _require(binary: str) -> None:
    if shutil.which(binary) is None:
        raise EnvironmentError(
            f"Required binary '{binary}' not found on PATH. Please install it."
        )


def _run(cmd: list[str]) -> None:
    log.debug("exec: %s", " ".join(str(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"stderr:\n{proc.stderr[-2000:]}"
        )


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        log.info("CUDA available -> %s", torch.cuda.get_device_name(0))
        return "cuda"
    log.warning("CUDA not available; falling back to CPU (this will be slow).")
    return "cpu"


# --------------------------------------------------------------------------- #
# Stage 1 — Download
# --------------------------------------------------------------------------- #
class AudioDownloader:
    """Downloads the best audio stream and decodes it to WAV via ffmpeg."""

    def __init__(self, workdir: Path):
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        _require("yt-dlp")
        _require("ffmpeg")

    def download(self, url: str, sample_rate: int = 44_100, stereo: bool = True) -> Path:
        wav = self.workdir / f"audio_{sample_rate}.wav"

        log.info("Downloading audio: %s", url)
        # Let yt-dlp own the extension via %(ext)s; the post-processor decides it.
        download_cmd = [
            "yt-dlp",
            "-f", "bestaudio/best",
            "--extract-audio",
            "--audio-format", "m4a",
            "--audio-quality", "0",
            "--no-playlist",
        ]
        cookies_file = os.environ.get("YT_DLP_COOKIES_FILE", "").strip()
        if cookies_file:
            cookies_path = Path(cookies_file)
            if not cookies_path.is_file():
                raise FileNotFoundError(
                    "YT_DLP_COOKIES_FILE does not point to a readable file: "
                    f"{cookies_path}"
                )
            download_cmd.extend(["--cookies", str(cookies_path)])
        download_cmd.extend([
            "-o", str(self.workdir / "download.%(ext)s"),
            url,
        ])
        _run(download_cmd)
        # Pick the produced media file (ignore .part / .ytdl leftovers).
        candidates = [
            p for p in self.workdir.glob("download.*")
            if p.suffix.lower() not in {".part", ".ytdl", ".temp"}
        ]
        if not candidates:
            raise FileNotFoundError("yt-dlp produced no audio file.")
        raw = max(candidates, key=lambda p: p.stat().st_size)

        log.info("Decoding to WAV (%d Hz, %s)", sample_rate, "stereo" if stereo else "mono")
        _run([
            "ffmpeg", "-y",
            "-i", str(raw),
            "-ac", "2" if stereo else "1",
            "-ar", str(sample_rate),
            "-vn",
            str(wav),
        ])
        return wav


# --------------------------------------------------------------------------- #
# Stage 2 — Source separation (Demucs / htdemucs)
# --------------------------------------------------------------------------- #
class VocalIsolator:
    """Isolates the vocal stem with Demucs htdemucs on the GPU."""

    def __init__(self, device: str, model_name: str = "htdemucs"):
        self.device = device
        self.model_name = model_name
        log.info("Loading Demucs model '%s' on %s", model_name, device)

        from demucs.pretrained import get_model

        self.model = get_model(model_name)
        self.model.to(device)
        self.model.eval()
        if "vocals" not in self.model.sources:
            raise RuntimeError(f"Model {model_name} has no 'vocals' source.")
        self.vocals_idx = self.model.sources.index("vocals")

    def isolate(self, audio_wav: Path, out_path: Path) -> Path:
        """Returns path to isolated vocals at the model's native sample rate."""
        import torch
        import torchaudio
        from demucs.apply import apply_model
        from demucs.audio import convert_audio, save_audio

        with torch.inference_mode():
            wav, sr = torchaudio.load(str(audio_wav))  # (channels, samples)
            # Match model expectations (channels + sample rate).
            wav = convert_audio(wav, sr, self.model.samplerate, self.model.audio_channels)

            # Normalise (Demucs was trained on loudness-normalised input).
            ref = wav.mean(0)
            mean, std = ref.mean(), ref.std() + 1e-8
            wav = (wav - mean) / std

            log.info("Separating stems on %s ...", self.device)
            sources = apply_model(
                self.model,
                # Keep the complete recording and Demucs' full-length output
                # accumulator in system RAM. apply_model still moves each
                # split to device for GPU inference. Putting the input on
                # CUDA makes Demucs allocate every output stem for the entire
                # recording on the GPU, which is unsafe with shared GPUs.
                # CPU accumulation trades abundant RAM for bounded VRAM.
                wav[None],
                device=self.device,
                split=True,          # chunked to bound VRAM
                overlap=0.25,
                progress=True,
            )[0]
            sources = sources * std + mean
            vocals = sources[self.vocals_idx].cpu()

        save_audio(vocals, str(out_path), samplerate=self.model.samplerate)
        log.info("Vocals written -> %s (%d Hz)", out_path, self.model.samplerate)
        return out_path


# --------------------------------------------------------------------------- #
# Stage 3 — ASR + forced alignment (WhisperX)
# --------------------------------------------------------------------------- #
class GPUTranscriber:
    """WhisperX large-v3 transcription with word-level forced alignment."""

    def __init__(
        self,
        device: str,
        model_size: str = "large-v3",
        compute_type: str = "float16",
        batch_size: int = 1,
        language: Optional[str] = None,
        alignment_device: Optional[str] = None,
    ):
        self.device = device
        self.alignment_device = alignment_device or device
        self.batch_size = batch_size
        self.language = language
        self._alignment_language: Optional[str] = None
        self._alignment_model = None
        self._alignment_metadata = None

        import whisperx

        self._whisperx = whisperx
        # faster-whisper backend requires int8 on CPU.
        if device == "cpu" and compute_type == "float16":
            log.warning("float16 unsupported on CPU; using int8.")
            compute_type = "int8"
        backend_device = device
        device_index = 0
        cuda_match = re.fullmatch(r"cuda(?::(\d+))?", device)
        if cuda_match:
            backend_device = "cuda"
            device_index = int(cuda_match.group(1) or 0)
        log.info(
            "Loading WhisperX '%s' (%s) on %s with Silero VAD; alignment on %s",
            model_size, compute_type, device, self.alignment_device,
        )
        self.model = whisperx.load_model(
            model_size,
            backend_device,
            device_index=device_index,
            compute_type=compute_type,
            language=language,
            vad_method="silero",
        )

    def _alignment_for_language(self, language: str):
        if self._alignment_language == language and self._alignment_model is not None:
            log.info("Reusing cached alignment model for language '%s'.", language)
            return self._alignment_model, self._alignment_metadata

        if self._alignment_model is not None:
            log.info(
                "Replacing cached alignment model for language '%s' with '%s'.",
                self._alignment_language,
                language,
            )
            self._alignment_model = None
            self._alignment_metadata = None
            if str(self.alignment_device).startswith("cuda"):
                import torch

                torch.cuda.empty_cache()

        log.info("Loading alignment model for language '%s' on %s ...", language, self.alignment_device)
        model, metadata = self._whisperx.load_align_model(
            language_code=language,
            device=self.alignment_device,
        )
        self._alignment_language = language
        self._alignment_model = model
        self._alignment_metadata = metadata
        return model, metadata

    def transcribe(self, vocals_16k: Path) -> list[dict]:
        wx = self._whisperx
        audio = wx.load_audio(str(vocals_16k))  # float32 mono @16k

        log.info("Transcribing ...")
        result = self.model.transcribe(
            audio, batch_size=self.batch_size, language=self.language
        )
        lang = result["language"]
        log.info("Detected language: %s", lang)

        log.info("Running forced alignment ...")
        align_model, metadata = self._alignment_for_language(lang)
        aligned = wx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self.alignment_device,
            return_char_alignments=False,
        )

        words: list[dict] = []
        for seg in aligned.get("segments", []):
            for w in seg.get("words", []):
                # Alignment occasionally fails for a token (numerals, symbols).
                if "start" not in w or "end" not in w:
                    continue
                token = str(w.get("word", "")).strip()
                if not token:
                    continue
                words.append(
                    {"word": token, "start": float(w["start"]), "end": float(w["end"])}
                )

        log.info("Aligned %d words.", len(words))
        return words


# --------------------------------------------------------------------------- #
# Stage 4 — Acoustic / prosodic features (Parselmouth)
# --------------------------------------------------------------------------- #
class AcousticAnalyzer:
    """
    Extracts F0-based prosody per word from the isolated vocal track.

    pitch_floor / pitch_ceiling default to 75-600 Hz (Praat default). For strongly
    exaggerated CDS, raise the ceiling (e.g. 800) so high-pitched peaks are not
    clipped by the pitch tracker.
    """

    def __init__(self, vocals_wav: Path, pitch_floor: float = 75.0, pitch_ceiling: float = 600.0):
        import parselmouth
        import numpy as np

        self._pm = parselmouth
        self._np = np
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling

        log.info(
            "Extracting F0 contour (%.0f-%.0f Hz) from %s",
            pitch_floor, pitch_ceiling, vocals_wav.name,
        )
        self.sound = parselmouth.Sound(str(vocals_wav))
        # Praat's pitch tracker expects mono; Demucs writes stereo vocals.
        if self.sound.n_channels > 1:
            self.sound = self.sound.convert_to_mono()
        # Autocorrelation pitch tracker (robust; Praat default).
        self.pitch = self.sound.to_pitch_ac(
            pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling
        )
        freqs = self.pitch.selected_array["frequency"]  # 0.0 == unvoiced
        self._times = self.pitch.xs()
        self._freqs = np.asarray(freqs, dtype=float)

    _VOWELS = "aeiouy"

    @classmethod
    def syllabify(cls, word: str) -> list[str]:
        """
        Split a word with a lightweight English orthographic heuristic.

        Vowel groups form nuclei. Between two nuclei, a single consonant starts
        the following syllable; a longer consonant cluster is split before its
        final consonant. A likely silent trailing "e" is not a separate nucleus.
        The returned spelling is useful for labeling analysis windows, but is not
        intended as a dictionary-quality phonetic syllabification.
        """
        letters = re.sub(r"[^A-Za-z]", "", word)
        if not letters:
            return [word or ""]

        lower = letters.lower()
        nuclei = list(re.finditer(r"[aeiouy]+", lower))
        if (
            len(nuclei) > 1
            and nuclei[-1].start() == len(lower) - 1
            and lower.endswith("e")
            and not lower.endswith(("le", "ee", "ie"))
        ):
            nuclei.pop()
        if len(nuclei) <= 1:
            return [letters]

        boundaries: list[int] = []
        for left, right in zip(nuclei, nuclei[1:]):
            consonant_count = right.start() - left.end()
            # With one consonant use V-CV; with a cluster use VC-CV.
            boundary = left.end() if consonant_count <= 1 else right.start() - 1
            boundaries.append(boundary)

        starts = [0, *boundaries]
        ends = [*boundaries, len(letters)]
        return [letters[start:end] for start, end in zip(starts, ends) if end > start]

    @classmethod
    def count_syllables(cls, word: str) -> int:
        return max(1, len(cls.syllabify(word)))

    @classmethod
    def make_syllable_units(cls, words: list[dict]) -> list[dict]:
        """Estimate equally sized syllable windows inside aligned word windows."""
        units: list[dict] = []
        for word in words:
            syllables = cls.syllabify(str(word["word"]))
            count = len(syllables)
            start = float(word["start"])
            end = float(word["end"])
            step = (end - start) / count if count else 0.0
            for index, syllable in enumerate(syllables):
                unit_start = start + index * step
                unit_end = end if index == count - 1 else start + (index + 1) * step
                units.append({
                    "syllable": syllable,
                    "parent_word": word["word"],
                    "syllable_index": index + 1,
                    "syllable_count": count,
                    "start": unit_start,
                    "end": unit_end,
                })
        return units

    def _pitch_in_window(self, start: float, end: float):
        mask = (self._times >= start) & (self._times <= end)
        seg = self._freqs[mask]
        return seg[seg > 0.0]  # keep only voiced frames

    def enrich(self, words: list[dict]) -> list[WordFeature]:
        features: list[WordFeature] = []
        n = len(words)
        for i, w in enumerate(words):
            start, end = w["start"], w["end"]
            duration = max(end - start, 1e-6)

            voiced = self._pitch_in_window(start, end)
            if voiced.size > 0:
                pitch_mean = round(float(self._np.mean(voiced)), 2)
                pitch_var = round(float(self._np.var(voiced)), 2)
            else:
                pitch_mean = None   # unvoiced word / silence -> null
                pitch_var = None

            if i < n - 1:
                pause = round(max(0.0, words[i + 1]["start"] - end), 3)
            else:
                pause = None

            syllables = self.count_syllables(w["word"])
            speech_rate = round(syllables / duration, 3)

            features.append(
                WordFeature(
                    word=w["word"],
                    start_time=round(start, 3),
                    end_time=round(end, 3),
                    pitch_mean_hz=pitch_mean,
                    pitch_variance=pitch_var,
                    pause_after_sec=pause,
                    speech_rate_sps=speech_rate,
                )
            )
        return features

    def enrich_syllables(self, words: list[dict]) -> list[SyllableFeature]:
        units = self.make_syllable_units(words)
        features: list[SyllableFeature] = []
        for i, unit in enumerate(units):
            start, end = unit["start"], unit["end"]
            duration = max(end - start, 1e-6)
            voiced = self._pitch_in_window(start, end)
            pitch_mean = round(float(self._np.mean(voiced)), 2) if voiced.size else None
            pitch_var = round(float(self._np.var(voiced)), 2) if voiced.size else None
            pause = (
                round(max(0.0, units[i + 1]["start"] - end), 3)
                if i < len(units) - 1 else None
            )
            features.append(SyllableFeature(
                syllable=unit["syllable"],
                parent_word=unit["parent_word"],
                syllable_index=unit["syllable_index"],
                syllable_count=unit["syllable_count"],
                start_time=round(start, 3),
                end_time=round(end, 3),
                pitch_mean_hz=pitch_mean,
                pitch_variance=pitch_var,
                pause_after_sec=pause,
                speech_rate_sps=round(1.0 / duration, 3),
            ))
        return features


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    url: str
    out_path: Path
    workdir: Path
    device: str = "auto"
    demucs_device: Optional[str] = None
    alignment_device: Optional[str] = None
    whisper_model: str = "large-v3"
    compute_type: str = "float16"
    demucs_model: str = "htdemucs"
    language: Optional[str] = None
    pitch_floor: float = 75.0
    pitch_ceiling: float = 600.0
    keep_workdir: bool = True
    asr_sr: int = 16_000
    granularity: str = "word"
    label: Optional[str] = None


class ProsodyPipeline:
    def __init__(self, cfg: PipelineConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = device or resolve_device(cfg.device)
        self.demucs_device = cfg.demucs_device or self.device
        self.alignment_device = cfg.alignment_device or self.device
        cfg.workdir.mkdir(parents=True, exist_ok=True)

    def _downsample_for_asr(self, vocals_hi: Path) -> Path:
        out = self.cfg.workdir / f"vocals_{self.cfg.asr_sr}_mono.wav"
        _run([
            "ffmpeg", "-y", "-i", str(vocals_hi),
            "-ac", "1", "-ar", str(self.cfg.asr_sr), "-vn", str(out),
        ])
        return out

    def run(
        self,
        isolator: Optional[VocalIsolator] = None,
        transcriber: Optional[GPUTranscriber] = None,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
    ) -> list[dict] | dict[str, list[dict]]:
        cfg = self.cfg

        def progress(stage: str, status: str, detail: str = "") -> None:
            log.info("stage=%s status=%s%s", stage, status, f" | {detail}" if detail else "")
            if progress_callback is not None:
                progress_callback(stage, status, detail)

        # 1. Download @44.1k stereo (Demucs-native)
        progress("download", "started", cfg.url)
        downloader = AudioDownloader(cfg.workdir)
        raw_wav = downloader.download(cfg.url, sample_rate=44_100, stereo=True)
        progress("download", "completed", raw_wav.name)

        # 2. Isolate vocals (stays @44.1k for pitch fidelity)
        progress("vocal_isolation", "started", cfg.demucs_model)
        isolator = isolator or VocalIsolator(
            self.demucs_device, model_name=cfg.demucs_model
        )
        vocals_hi = isolator.isolate(raw_wav, cfg.workdir / "vocals_44100.wav")
        progress("vocal_isolation", "completed", vocals_hi.name)

        # 3. ASR + alignment on a 16k mono copy of the ISOLATED vocals
        progress("asr_conversion", "started", f"{cfg.asr_sr} Hz mono")
        vocals_16k = self._downsample_for_asr(vocals_hi)
        progress("asr_conversion", "completed", vocals_16k.name)
        progress("transcription_alignment", "started", cfg.whisper_model)
        transcriber = transcriber or GPUTranscriber(
            self.device,
            model_size=cfg.whisper_model,
            compute_type=cfg.compute_type,
            language=cfg.language,
            alignment_device=self.alignment_device,
        )
        words = transcriber.transcribe(vocals_16k)
        progress("transcription_alignment", "completed", f"{len(words)} aligned words")
        if not words:
            log.warning("No words were transcribed/aligned; output will be empty.")

        # 4. Prosody from the high-res isolated vocals
        progress("prosody_analysis", "started", cfg.granularity)
        analyzer = AcousticAnalyzer(
            vocals_hi, pitch_floor=cfg.pitch_floor, pitch_ceiling=cfg.pitch_ceiling
        )
        if cfg.granularity == "word":
            word_payload = [f.to_dict() for f in analyzer.enrich(words)]
            payload: list[dict] | dict[str, list[dict]] = word_payload
            record_count = len(word_payload)
        elif cfg.granularity == "syllable":
            syllable_payload = [f.to_dict() for f in analyzer.enrich_syllables(words)]
            payload = syllable_payload
            record_count = len(syllable_payload)
        else:
            word_payload = [f.to_dict() for f in analyzer.enrich(words)]
            syllable_payload = [f.to_dict() for f in analyzer.enrich_syllables(words)]
            payload = {"words": word_payload, "syllables": syllable_payload}
            record_count = len(word_payload) + len(syllable_payload)
        if cfg.label is not None:
            collections = (
                {"words": payload} if cfg.granularity == "word"
                else {"syllables": payload} if cfg.granularity == "syllable"
                else payload
            )
            payload = {
                "label": cfg.label,
                "source_url": cfg.url,
                "language": cfg.language,
                "granularity": cfg.granularity,
                **collections,
            }
        progress("prosody_analysis", "completed", f"{record_count} records")
        progress("save_output", "started", str(cfg.out_path))
        cfg.out_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        log.info("Wrote %d %s records -> %s", record_count, cfg.granularity, cfg.out_path)
        progress("save_output", "completed", str(cfg.out_path))

        if not cfg.keep_workdir:
            progress("cleanup", "started", str(cfg.workdir))
            shutil.rmtree(cfg.workdir)
            progress("cleanup", "completed", str(cfg.workdir))
        return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[list[str]] = None) -> PipelineConfig:
    p = argparse.ArgumentParser(description="CDS prosody extraction pipeline.")
    p.add_argument("url", help="YouTube URL")
    p.add_argument("--out", type=Path, default=Path("output.json"))
    p.add_argument("--workdir", type=Path, default=Path("./_pipeline_run"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--whisper-model", default="large-v3")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--demucs-model", default="htdemucs")
    p.add_argument("--demucs-device", default=None,
                   help="Device for Demucs (default: same as --device).")
    p.add_argument("--alignment-device", default=None,
                   help="Device for forced alignment (default: same as --device).")
    p.add_argument("--language", default=None, help="Force language code (e.g. 'en').")
    p.add_argument("--pitch-floor", type=float, default=75.0)
    p.add_argument("--pitch-ceiling", type=float, default=600.0,
                   help="Raise (e.g. 800) for exaggerated CDS to avoid clipping.")
    p.add_argument("--granularity", choices=["word", "syllable", "both"], default="word",
                   help="Write word records, estimated syllable records, or both.")
    p.add_argument("--label", choices=["kids", "normal"], default=None,
                   help="Optional dataset label included in the output JSON.")
    p.add_argument("--clean", action="store_true", help="Delete workdir afterwards.")
    a = p.parse_args(argv)
    return PipelineConfig(
        url=a.url, out_path=a.out, workdir=a.workdir, device=a.device,
        demucs_device=a.demucs_device, alignment_device=a.alignment_device,
        whisper_model=a.whisper_model, compute_type=a.compute_type,
        demucs_model=a.demucs_model, language=a.language,
        pitch_floor=a.pitch_floor, pitch_ceiling=a.pitch_ceiling,
        keep_workdir=not a.clean, granularity=a.granularity, label=a.label,
    )


def main() -> int:
    cfg = parse_args()
    try:
        ProsodyPipeline(cfg).run()
    except Exception as exc:  # top-level guardrail
        log.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
