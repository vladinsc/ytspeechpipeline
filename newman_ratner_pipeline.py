"""Prepare Newman/Ratner CHAT data for transcript-guided prosody research.

The default alignment route uses WhisperX word alignment with supplied CHAT
text and CHAT time links.  A separate Silero-VAD WhisperX transcription run can
be written beside it to test the annotation without changing the annotation.
MFA remains available only for a legacy, explicitly selected comparison route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from acoustic_features import summarize_sound
from mfa_alignment import MFAAligner
from speech_pipeline import AcousticAnalyzer, GPUTranscriber


LOG = logging.getLogger("newman-ratner")
TIME_LINK = re.compile(r"\x15(\d+)_(\d+)\x15")
SPEAKER_LINE = re.compile(r"^\*([A-Za-z0-9]+):\s*(.*)$")
RETRACED = re.compile(r"<[^>]*>\s*\[/{1,2}\]")
UNCERTAIN = re.compile(r"\b(?:xxx|yyy|www)\b", re.IGNORECASE)
AGE_CHOICES = ("07", "10", "11", "18", "24")


@dataclass(frozen=True)
class ChatTurn:
    utterance_id: str
    speaker: str
    start_sec: float
    end_sec: float
    transcript: str
    chat_text: str
    dependent_tiers: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)


def clean_chat_text(text: str) -> str | None:
    """Make a conservative MFA transcript; reject unintelligible turns."""
    text = TIME_LINK.sub("", text)
    if UNCERTAIN.search(text):
        return None
    text = RETRACED.sub(" ", text)
    text = re.sub(r"&=[^\s]+", " ", text)  # non-speech events
    text = re.sub(r"&-([A-Za-z]+)", r"\1", text)  # audible fillers
    text = re.sub(r"\[[^\]]*\]", " ", text)  # CHAT comments and codes
    text = re.sub(r"\(([^)]*)\)", r"\1", text)  # e.g. (be)cause
    text = text.replace("_", " ").replace("<", "").replace(">", "")
    text = re.sub(r"\+[/!?.]+", " ", text)
    text = re.sub(r"[^A-Za-z0-9'\-\s.,?!]", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,!?:;-\t")
    if not text or text == "0" or not re.search(r"[A-Za-z]", text):
        return None
    return text


def parse_chat(path: Path) -> tuple[list[str], list[dict]]:
    """Preserve headers, all speaker turns, and dependent CHAT tiers in JSON."""
    if not path.is_file():
        raise FileNotFoundError(path)
    headers: list[str] = []
    raw_turns: list[dict] = []
    current: dict | None = None
    active_tier = "main"

    def flush() -> None:
        nonlocal current
        if current is not None:
            raw_turns.append(current)
            current = None

    for line in path.read_text(encoding="utf-8-sig").splitlines():
        speaker = SPEAKER_LINE.match(line)
        dependent = re.match(r"^%([^:]+):\s*(.*)$", line)
        if speaker:
            flush()
            current = {"speaker": speaker.group(1).upper(), "main": [speaker.group(2)],
                       "tiers": {}}
            active_tier = "main"
        elif dependent and current is not None:
            active_tier = dependent.group(1).lower()
            current["tiers"].setdefault(active_tier, []).append(dependent.group(2))
        elif current is not None and line.startswith(("\t", " ")):
            if active_tier == "main":
                current["main"].append(line.strip())
            elif active_tier in current["tiers"]:
                current["tiers"][active_tier][-1] += " " + line.strip()
        else:
            flush()
            if line.startswith("@"):
                headers.append(line)
            active_tier = "main"
    flush()

    turns = []
    for index, raw in enumerate(raw_turns):
        chat_text = " ".join(raw["main"])
        links = [(int(a), int(b)) for a, b in TIME_LINK.findall(chat_text)]
        start_sec = links[0][0] / 1000 if len(links) == 1 else None
        end_sec = links[0][1] / 1000 if len(links) == 1 else None
        turns.append({
            "utterance_id": f"utt_{index:05d}",
            "speaker": raw["speaker"],
            "chat_text": TIME_LINK.sub("", chat_text).strip(),
            "transcript": clean_chat_text(chat_text),
            "time_links_ms": links,
            "start_sec": start_sec, "end_sec": end_sec,
            "dependent_tiers": {
                name: " ".join(values) for name, values in raw["tiers"].items()
            },
        })
    return headers, turns


def _speaker_turns(path: Path):
    for row in parse_chat(path)[1]:
        yield row["speaker"], row["chat_text"]


def convert_chat(
    path: Path, *, speaker: str = "MOT", max_duration_sec: float = 25.0,
    max_words_per_sec: float = 8.0,
) -> tuple[list[ChatTurn], dict[str, int]]:
    """Return timed, clean turns and reasons other turns were excluded."""
    if not path.is_file():
        raise FileNotFoundError(path)
    counts: Counter[str] = Counter()
    turns: list[ChatTurn] = []
    speaker = speaker.upper()
    for row in parse_chat(path)[1]:
        tier = row["speaker"]
        counts["speaker_tiers"] += 1
        if tier != speaker:
            counts["other_speaker"] += 1
            continue
        links = row["time_links_ms"]
        if len(links) != 1:
            counts["missing_or_multiple_time_links"] += 1
            continue
        start_ms, end_ms = links[0]
        if not 0 <= start_ms < end_ms:
            counts["invalid_times"] += 1
            continue
        if (end_ms - start_ms) / 1000 > max_duration_sec:
            counts["over_duration_limit"] += 1
            continue
        transcript = row["transcript"]
        if transcript is None:
            counts["uncertain_or_nonlexical"] += 1
            continue
        if len(transcript.split()) / ((end_ms - start_ms) / 1000) > max_words_per_sec:
            counts["implausible_word_rate"] += 1
            continue
        turns.append(ChatTurn(
            utterance_id=row["utterance_id"], speaker=tier,
            start_sec=start_ms / 1000, end_sec=end_ms / 1000,
            transcript=transcript, chat_text=row["chat_text"],
            dependent_tiers=row["dependent_tiers"],
        ))
    counts["selected"] = len(turns)
    return sorted(turns, key=lambda item: (item.start_sec, item.utterance_id)), dict(counts)


def chat_text_overlap(first: Path, second: Path) -> float:
    """Jaccard overlap of tier text, ignoring media timestamps."""
    def lines(path: Path) -> set[tuple[str, str]]:
        return {
            (speaker, cleaned.lower())
            for speaker, original in _speaker_turns(path)
            if (cleaned := clean_chat_text(original))
        }

    left, right = lines(first), lines(second)
    return len(left & right) / len(left | right) if left or right else 0.0


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {result.stderr[-1200:]}")


def _audio_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Cannot probe {path}: {result.stderr[-1200:]}")
    return float(result.stdout.strip())


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    completed.add(json.loads(line)["utterance_id"])
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"Invalid result JSONL at {path}:{line_number}") from exc
    return completed


def _check_resume_config(
    output_jsonl: Path, *, turns: list[ChatTurn], audio: Path,
    dictionary: str, acoustic_model: str, pitch_floor: float,
    pitch_ceiling: float, metadata: dict | None,
    transcript_source: str = "CHAT", mfa_config: Path | None = None,
    alignment_backend: str = "mfa", whisper_model: str | None = None,
    compute_type: str | None = None, alignment_device: str | None = None,
    chat_context_sec: float | None = None,
) -> None:
    """Refuse to mix results from different transcripts, audio, or settings."""
    audio_stat = audio.stat()
    config = {
        "schema_version": "newman_ratner_features_v5",
        "source_audio": str(audio.resolve()),
        "source_audio_size": audio_stat.st_size,
        "source_audio_mtime_ns": audio_stat.st_mtime_ns,
        "turns_sha256": hashlib.sha256(json.dumps(
            [asdict(turn) for turn in turns], sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest(),
        "dictionary": dictionary, "acoustic_model": acoustic_model,
        "mfa_config": (str(mfa_config.resolve()) if mfa_config else None),
        "mfa_config_sha256": (
            hashlib.sha256(mfa_config.read_bytes()).hexdigest() if mfa_config else None
        ),
        "alignment_backend": alignment_backend,
        "whisper_model": whisper_model,
        "compute_type": compute_type,
        "alignment_device": alignment_device,
        "chat_context_sec": chat_context_sec,
        "pitch_floor": pitch_floor, "pitch_ceiling": pitch_ceiling,
        "transcript_source": transcript_source,
        "metadata": metadata or {},
    }
    config_path = output_jsonl.with_suffix(output_jsonl.suffix + ".config.json")
    if output_jsonl.exists() and output_jsonl.stat().st_size and not config_path.is_file():
        raise ValueError(f"Cannot resume {output_jsonl} without {config_path}")
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(
                f"Input or settings changed for {output_jsonl}. Choose a new output directory."
            )
    else:
        _write_json(config_path, config)


def process_turns(
    turns: list[ChatTurn], audio: Path, output_jsonl: Path,
    *, batch_size: int = 8, max_utterances: int = 0,
    dictionary: str = "english_us_mfa", acoustic_model: str = "english_mfa",
    conda_env: str | None = None, pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0,
    metadata: dict | None = None,
    transcript_source: str = "CHAT", mfa_config: Path | None = None,
) -> dict[str, int]:
    """Align a few short clips at a time; append one feature JSON per turn."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not audio.is_file():
        raise FileNotFoundError(audio)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _check_resume_config(
        output_jsonl, turns=turns, audio=audio, dictionary=dictionary,
        acoustic_model=acoustic_model, pitch_floor=pitch_floor,
        pitch_ceiling=pitch_ceiling, metadata=metadata,
        transcript_source=transcript_source, mfa_config=mfa_config,
    )
    aligner = MFAAligner(dictionary, acoustic_model, conda_env=conda_env,
                         config_path=mfa_config)
    aligner.ensure_available()
    duration = _audio_duration(audio)
    eligible = [turn for turn in turns if turn.end_sec <= duration + 0.05]
    out_of_bounds = len(turns) - len(eligible)
    if max_utterances:
        eligible = eligible[:max_utterances]
    done = _completed_ids(output_jsonl)
    pending = [turn for turn in eligible if turn.utterance_id not in done]
    completed_now = 0

    for first in range(0, len(pending), batch_size):
        batch = pending[first:first + batch_size]
        # The temporary directory and every clip stay bounded by batch_size.
        with tempfile.TemporaryDirectory(prefix="newman_batch_", dir=output_jsonl.parent) as temp:
            batch_root = Path(temp)
            if not batch_root.resolve().is_relative_to(output_jsonl.parent.resolve()):
                raise RuntimeError("Temporary batch escaped the output directory")
            corpus = batch_root / "mfa_input"
            analysis = batch_root / "analysis"
            corpus.mkdir()
            analysis.mkdir()
            for turn in batch:
                clip = analysis / f"{turn.utterance_id}.wav"
                _run(["ffmpeg", "-y", "-ss", str(turn.start_sec), "-i", str(audio),
                      "-t", str(turn.end_sec - turn.start_sec), "-ac", "1", "-vn", str(clip)])
                _run(["ffmpeg", "-y", "-i", str(clip), "-ac", "1", "-ar", "16000",
                      str(corpus / f"{turn.utterance_id}.wav")])
                (corpus / f"{turn.utterance_id}.lab").write_text(
                    turn.transcript, encoding="utf-8"
                )
            aligned = aligner.align_corpus(corpus, batch_root / "mfa_output")
            records = []
            for turn in batch:
                alignment = aligned[turn.utterance_id]
                analyzer = AcousticAnalyzer(
                    analysis / f"{turn.utterance_id}.wav",
                    pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling,
                )
                words = [feature.to_dict() for feature in analyzer.enrich(alignment.words)]
                syllables = [feature.to_dict() for feature in analyzer.enrich_syllables(alignment.words)]
                acoustic_features = summarize_sound(
                    analyzer.sound, analyzer.pitch, alignment.words
                )
                for item in (*words, *syllables):
                    item["start_time"] = round(item["start_time"] + turn.start_sec, 3)
                    item["end_time"] = round(item["end_time"] + turn.start_sec, 3)
                phones = [
                    {**phone, "start": round(phone["start"] + turn.start_sec, 3),
                     "end": round(phone["end"] + turn.start_sec, 3)}
                    for phone in alignment.phones
                ]
                reference_tokens = len(turn.transcript.split())
                aligned_tokens = len(words)
                aligned_ratio = aligned_tokens / reference_tokens if reference_tokens else 0.0
                quality_flags = []
                if aligned_ratio < 0.6 or aligned_ratio > 1.4:
                    quality_flags.append("alignment_word_count_mismatch")
                if any(
                    word["start"] < -0.05 or word["end"] > turn.end_sec - turn.start_sec + 0.1
                    for word in alignment.words
                ):
                    quality_flags.append("alignment_outside_clip")
                records.append({
                    "schema_version": "newman_ratner_features_v5",
                    **(metadata or {}),
                    "utterance_id": turn.utterance_id,
                    "speaker": turn.speaker,
                    "speaker_attribution": "CHAT_speaker_tier",
                    "source": str(audio.resolve()),
                    "source_audio": str(audio.resolve()),
                    "language": "en", "granularity": "both",
                    "start_sec": turn.start_sec, "end_sec": turn.end_sec,
                    "transcript": turn.transcript,
                    "dependent_tiers": turn.dependent_tiers,
                    "annotations": turn.annotations,
                    "transcript_source": transcript_source,
                    "alignment_method": "mfa",
                    "phone_set": dictionary,
                    "alignment_qc": {
                        "reference_token_count": reference_tokens,
                        "aligned_word_count": aligned_tokens,
                        "aligned_word_ratio": round(aligned_ratio, 3),
                        "quality_flags": quality_flags,
                    },
                    "words": words, "syllables": syllables, "phones": phones,
                    "acoustic_features": acoustic_features,
                })
            with output_jsonl.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            completed_now += len(records)
            LOG.info("%s: completed %d/%d new turns", audio.name, completed_now, len(pending))
    return {
        "eligible": len(eligible), "already_completed": len(done & {t.utterance_id for t in eligible}),
        "completed_now": completed_now, "out_of_audio_bounds": out_of_bounds,
    }


def _clip_audio(audio: Path, output: Path, start_sec: float, end_sec: float, *, sample_rate: int | None = None) -> None:
    """Make a short mono clip without decoding an entire corpus recording."""
    command = [
        "ffmpeg", "-y", "-ss", str(start_sec), "-i", str(audio),
        "-t", str(end_sec - start_sec), "-ac", "1", "-vn",
    ]
    if sample_rate is not None:
        command.extend(["-ar", str(sample_rate)])
    command.append(str(output))
    _run(command)


def _feature_payload(
    analyzer: AcousticAnalyzer, words: list[dict], *, clip_start_sec: float,
    chat_start_local: float, chat_end_local: float,
) -> tuple[list[dict], list[dict], dict]:
    """Extract word/syllable prosody in a padded clip, summarize CHAT interval."""
    word_payload = [item.to_dict() for item in analyzer.enrich(words)]
    syllable_payload = [item.to_dict() for item in analyzer.enrich_syllables(words)]
    acoustic_features = summarize_sound(
        analyzer.sound, analyzer.pitch, words,
        window_start=chat_start_local, window_end=chat_end_local,
    )
    for item in (*word_payload, *syllable_payload):
        item["start_time"] = round(item["start_time"] + clip_start_sec, 3)
        item["end_time"] = round(item["end_time"] + clip_start_sec, 3)
    return word_payload, syllable_payload, acoustic_features


def process_whisperx_chat_turns(
    turns: list[ChatTurn], audio: Path, output_jsonl: Path,
    *, transcriber: GPUTranscriber, batch_size: int = 8,
    max_utterances: int = 0, pitch_floor: float = 75.0,
    pitch_ceiling: float = 600.0, chat_context_sec: float = 0.25,
    whisper_model: str = "large-v3", compute_type: str = "float16",
    alignment_device: str | None = None, metadata: dict | None = None,
) -> dict[str, int]:
    """Align trusted CHAT text with WhisperX and extract word-level prosody.

    CHAT links define a search window, not the final word boundaries.  Each
    alignment clip includes a small amount of audio context so a word at an
    annotation edge is not truncated.  Phone boundaries are intentionally not
    emitted because WhisperX exposes word, rather than phone, timestamps.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if chat_context_sec < 0:
        raise ValueError("chat_context_sec cannot be negative")
    if not audio.is_file():
        raise FileNotFoundError(audio)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    _check_resume_config(
        output_jsonl, turns=turns, audio=audio,
        dictionary="not_applicable", acoustic_model="not_applicable",
        pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling, metadata=metadata,
        transcript_source="CHAT", alignment_backend="whisperx_chat_guided",
        whisper_model=whisper_model, compute_type=compute_type,
        alignment_device=alignment_device, chat_context_sec=chat_context_sec,
    )
    duration = _audio_duration(audio)
    eligible = [turn for turn in turns if turn.end_sec <= duration + 0.05]
    out_of_bounds = len(turns) - len(eligible)
    if max_utterances:
        eligible = eligible[:max_utterances]
    done = _completed_ids(output_jsonl)
    pending = [turn for turn in eligible if turn.utterance_id not in done]
    completed_now = 0

    for first in range(0, len(pending), batch_size):
        batch = pending[first:first + batch_size]
        with tempfile.TemporaryDirectory(prefix="newman_whisperx_", dir=output_jsonl.parent) as temp:
            batch_root = Path(temp)
            if not batch_root.resolve().is_relative_to(output_jsonl.parent.resolve()):
                raise RuntimeError("Temporary batch escaped the output directory")
            records = []
            for turn in batch:
                clip_start = max(0.0, turn.start_sec - chat_context_sec)
                clip_end = min(duration, turn.end_sec + chat_context_sec)
                chat_start_local = turn.start_sec - clip_start
                chat_end_local = turn.end_sec - clip_start
                analysis_clip = batch_root / f"{turn.utterance_id}_analysis.wav"
                alignment_clip = batch_root / f"{turn.utterance_id}_alignment.wav"
                _clip_audio(audio, analysis_clip, clip_start, clip_end)
                _clip_audio(audio, alignment_clip, clip_start, clip_end, sample_rate=16_000)
                aligned_words = transcriber.align_known_transcript(
                    alignment_clip, turn.transcript, language="en",
                    start_sec=0.0, end_sec=clip_end - clip_start,
                )
                analyzer = AcousticAnalyzer(
                    analysis_clip, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling,
                )
                words, syllables, acoustic_features = _feature_payload(
                    analyzer, aligned_words, clip_start_sec=clip_start,
                    chat_start_local=chat_start_local, chat_end_local=chat_end_local,
                )
                reference_tokens = len(turn.transcript.split())
                aligned_tokens = len(words)
                aligned_ratio = aligned_tokens / reference_tokens if reference_tokens else 0.0
                edge_tolerance = 0.15
                quality_flags = []
                if aligned_ratio < 0.6 or aligned_ratio > 1.4:
                    quality_flags.append("alignment_word_count_mismatch")
                if any(
                    item["start_time"] < turn.start_sec - edge_tolerance
                    or item["end_time"] > turn.end_sec + edge_tolerance
                    for item in words
                ):
                    quality_flags.append("word_outside_CHAT_time_link")
                records.append({
                    "schema_version": "newman_ratner_features_v5",
                    **(metadata or {}),
                    "utterance_id": turn.utterance_id,
                    "speaker": turn.speaker,
                    "speaker_attribution": "CHAT_speaker_tier",
                    "source": str(audio.resolve()), "source_audio": str(audio.resolve()),
                    "language": "en", "granularity": "both",
                    "start_sec": turn.start_sec, "end_sec": turn.end_sec,
                    "chat_time_link_sec": {"start": turn.start_sec, "end": turn.end_sec},
                    "alignment_search_window_sec": {"start": clip_start, "end": clip_end},
                    "transcript": turn.transcript, "chat_text": turn.chat_text,
                    "dependent_tiers": turn.dependent_tiers, "annotations": turn.annotations,
                    "transcript_source": "CHAT",
                    "alignment_method": "whisperx_chat_guided",
                    "alignment_scope": "word_only",
                    "phone_set": None, "phones": [],
                    "alignment_qc": {
                        "reference_token_count": reference_tokens,
                        "aligned_word_count": aligned_tokens,
                        "aligned_word_ratio": round(aligned_ratio, 3),
                        "quality_flags": quality_flags,
                    },
                    "words": words, "syllables": syllables,
                    "acoustic_features": acoustic_features,
                })
            with output_jsonl.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
            completed_now += len(records)
            LOG.info("%s: completed %d/%d CHAT-guided WhisperX turns", audio.name, completed_now, len(pending))
    return {
        "eligible": len(eligible),
        "already_completed": len(done & {turn.utterance_id for turn in eligible}),
        "completed_now": completed_now,
        "out_of_audio_bounds": out_of_bounds,
    }


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def _word_error_rate(reference: str, hypothesis: str) -> dict[str, int | float]:
    """Small dependency-free WER for diagnostic annotation checks."""
    ref, hyp = _normalized_tokens(reference), _normalized_tokens(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, reference_word in enumerate(ref, 1):
        current = [i]
        for j, hypothesis_word in enumerate(hyp, 1):
            current.append(min(
                previous[j] + 1, current[j - 1] + 1,
                previous[j - 1] + (reference_word != hypothesis_word),
            ))
        previous = current
    errors = previous[-1]
    return {
        "reference_token_count": len(ref), "hypothesis_token_count": len(hyp),
        "edit_distance": errors,
        "wer": round(errors / len(ref), 3) if ref else (0.0 if not hyp else 1.0),
    }


def _interval_overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def validate_independent_asr(
    turns: list[ChatTurn], audio: Path, output_json: Path, *,
    transcriber: GPUTranscriber, whisper_model: str = "large-v3",
    compute_type: str = "float16", alignment_device: str | None = None,
    metadata: dict | None = None,
) -> dict[str, int | float]:
    """Write independent Silero-VAD ASR output and CHAT comparison diagnostics."""
    if not audio.is_file():
        raise FileNotFoundError(audio)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    audio_stat = audio.stat()
    config = {
        "schema_version": "newman_ratner_asr_validation_v1",
        "source_audio": str(audio.resolve()), "source_audio_size": audio_stat.st_size,
        "source_audio_mtime_ns": audio_stat.st_mtime_ns,
        "turns_sha256": hashlib.sha256(json.dumps(
            [asdict(turn) for turn in turns], sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")).hexdigest(),
        "whisper_model": whisper_model, "compute_type": compute_type,
        "alignment_device": alignment_device, "metadata": metadata or {},
    }
    config_path = output_json.with_suffix(output_json.suffix + ".config.json")
    if output_json.exists() and not config_path.is_file():
        raise ValueError(f"Cannot replace {output_json} without {config_path}")
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError(f"Input or settings changed for {output_json}. Choose a new output directory.")
    else:
        _write_json(config_path, config)

    with tempfile.TemporaryDirectory(prefix="newman_independent_asr_", dir=output_json.parent) as temp:
        audio_16k = Path(temp) / "source_16k.wav"
        _clip_audio(audio, audio_16k, 0.0, _audio_duration(audio), sample_rate=16_000)
        asr = transcriber.transcribe_with_alignment(audio_16k)
    checks = []
    matched_segment_indexes = set()
    for turn in turns:
        matches = [
            (index, segment) for index, segment in enumerate(asr["segments"])
            if _interval_overlap(turn.start_sec, turn.end_sec, segment["start"], segment["end"]) > 0
        ]
        matched_segment_indexes.update(index for index, _segment in matches)
        hypothesis = " ".join(segment["text"] for _index, segment in matches)
        checks.append({
            "utterance_id": turn.utterance_id,
            "chat_time_link_sec": {"start": turn.start_sec, "end": turn.end_sec},
            "chat_transcript": turn.transcript,
            "overlapping_asr_segment_count": len(matches),
            "asr_transcript": hypothesis,
            "text_comparison": _word_error_rate(turn.transcript, hypothesis),
        })
    combined_chat = " ".join(turn.transcript for turn in turns)
    combined_asr = " ".join(asr["segments"][index]["text"] for index in sorted(matched_segment_indexes))
    payload = {
        "schema_version": "newman_ratner_asr_validation_v1",
        **(metadata or {}),
        "source_audio": str(audio.resolve()),
        "transcript_source": "whisperx_asr_silero_vad_independent",
        "validation_purpose": "diagnostic_only_does_not_modify_CHAT_annotation",
        "language": asr["language"], "asr_segments": asr["segments"], "asr_words": asr["words"],
        "chat_turn_checks": checks,
        "target_speaker_overlap_summary": _word_error_rate(combined_chat, combined_asr),
    }
    _write_json(output_json, payload)
    summary = payload["target_speaker_overlap_summary"]
    LOG.info("%s: wrote independent Silero-VAD ASR validation (%d segments, WER=%s)",
             audio.name, len(asr["segments"]), summary["wer"])
    return {"asr_segments": len(asr["segments"]), "asr_words": len(asr["words"]),
            "target_speaker_wer": summary["wer"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--age", choices=AGE_CHOICES, required=True)
    parser.add_argument("--recording", required=True, help="Recording stem, e.g. 4269LP")
    parser.add_argument("--root", type=Path, default=Path("data/staging/newman_ratner"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/newman_ratner"))
    parser.add_argument("--condition", choices=("play", "interview", "both"), default="both")
    parser.add_argument("--speaker", default="MOT", help="CHAT speaker tier (default: MOT)")
    parser.add_argument("--max-duration-sec", type=float, default=25.0)
    parser.add_argument("--max-words-per-sec", type=float, default=8.0,
                        help="Reject implausible CHAT time links (default: 8)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-utterances", type=int, default=0,
                        help="Pilot limit per condition; 0 processes all eligible turns")
    parser.add_argument("--alignment-backend", choices=("whisperx_chat", "mfa"),
                        default="whisperx_chat",
                        help="CHAT-guided WhisperX word alignment is the default; MFA is legacy.")
    parser.add_argument("--device", default="cuda",
                        help="WhisperX ASR device, normally cuda or cuda:0 on Bengee.")
    parser.add_argument("--alignment-device", default=None,
                        help="WhisperX CTC alignment device (default: --device).")
    parser.add_argument("--whisper-model", default="large-v3",
                        help="WhisperX ASR model for independent Silero-VAD validation.")
    parser.add_argument("--compute-type", default="float16",
                        help="WhisperX compute type, e.g. float16 on CUDA or int8 on CPU.")
    parser.add_argument("--chat-context-sec", type=float, default=0.25,
                        help="Audio context before/after each CHAT link for word alignment.")
    parser.add_argument("--validate-independent-asr", action="store_true",
                        help="Run full-audio Silero-VAD WhisperX ASR and compare it with CHAT; writes a separate diagnostic JSON.")
    parser.add_argument("--mfa-conda-env", default=None)
    parser.add_argument("--mfa-dictionary", default="english_us_mfa")
    parser.add_argument("--mfa-acoustic-model", default="english_mfa")
    parser.add_argument("--mfa-config", type=Path,
                        help="Optional MFA YAML/JSON configuration, such as wider decoding beams")
    parser.add_argument("--pitch-floor", type=float, default=75.0)
    parser.add_argument("--pitch-ceiling", type=float, default=600.0)
    parser.add_argument("--align", action="store_true",
                        help="Align supplied CHAT text and extract prosody; without this, only convert CHAT to JSON")
    args = parser.parse_args(argv)
    if (args.max_duration_sec <= 0 or args.max_words_per_sec <= 0
            or args.batch_size <= 0 or args.max_utterances < 0
            or args.chat_context_sec < 0):
        parser.error("duration and batch size must be positive; max utterances cannot be negative")
    recording = Path(args.recording).stem
    if not re.fullmatch(r"[A-Za-z0-9_-]+", recording):
        parser.error("recording must be a simple file stem")
    conditions = ("play", "interview") if args.condition == "both" else (args.condition,)
    base = args.out_dir / args.age / recording
    main_chat = args.root / "transcripts" / "extracted" / args.age / f"{recording}.cha"
    interview_chat = args.root / "transcripts" / "extracted" / "Interviews" / args.age / f"{recording}.cha"
    overlap = (chat_text_overlap(main_chat, interview_chat)
               if main_chat.is_file() and interview_chat.is_file() else None)
    contrast_status = (
        "near_duplicate_CHAT_text_do_not_compare" if overlap is not None and overlap >= 0.8
        else "requires_context_and_audio_review"
    )
    if overlap is not None and overlap >= 0.8:
        LOG.warning("%s: play/interview CHAT text overlaps %.1f%%; do not treat as independent CDS/ADS", recording, overlap * 100)
    transcriber = None
    if (args.align and args.alignment_backend == "whisperx_chat") or args.validate_independent_asr:
        transcriber = GPUTranscriber(
            args.device, model_size=args.whisper_model, compute_type=args.compute_type,
            language="en", alignment_device=args.alignment_device,
        )
    for condition in conditions:
        relative = Path(args.age) / recording
        if condition == "play":
            chat = args.root / "transcripts" / "extracted" / relative.with_suffix(".cha")
            audio = args.root / "media" / relative.with_suffix(".wav")
        else:
            chat = args.root / "transcripts" / "extracted" / "Interviews" / relative.with_suffix(".cha")
            audio = args.root / "media" / "Interviews" / relative.with_suffix(".mp3")
        if not audio.is_file():
            raise FileNotFoundError(audio)
        headers, all_turns = parse_chat(chat)
        turns, counts = convert_chat(chat, speaker=args.speaker,
                                     max_duration_sec=args.max_duration_sec,
                                     max_words_per_sec=args.max_words_per_sec)
        transcript_json = {
            "schema_version": "newman_ratner_transcript_v1",
            "corpus": "NewmanRatner", "age_group": args.age,
            "recording_id": recording, "condition": condition,
            "condition_basis": "media_folder_only_not_verified_from_audio",
            "pair_chat_text_overlap": overlap,
            "paired_contrast_status": contrast_status,
            "speaker": args.speaker.upper(), "source_audio": str(audio.resolve()),
            "source_chat": str(chat.resolve()), "conversion_counts": counts,
            "headers": headers, "all_turns": all_turns,
            "utterances": [asdict(turn) for turn in turns],
        }
        _write_json(base / f"{condition}_transcript.json", transcript_json)
        LOG.info("%s %s: %d timed %s turns", recording, condition, len(turns), args.speaker)
        metadata = {
            "age_group": args.age, "recording_id": recording,
            "source_section": condition,
            "condition_basis": "media_folder_only_not_verified_from_audio",
            "pair_chat_text_overlap": overlap,
            "paired_contrast_status": contrast_status,
        }
        status = {}
        if args.align:
            if args.alignment_backend == "whisperx_chat":
                status["chat_guided_alignment"] = process_whisperx_chat_turns(
                    turns, audio, base / f"{condition}_whisperx_chat_features.jsonl",
                    transcriber=transcriber, batch_size=args.batch_size,
                    max_utterances=args.max_utterances,
                    pitch_floor=args.pitch_floor, pitch_ceiling=args.pitch_ceiling,
                    chat_context_sec=args.chat_context_sec,
                    whisper_model=args.whisper_model, compute_type=args.compute_type,
                    alignment_device=args.alignment_device, metadata=metadata,
                )
            else:
                status["legacy_mfa_alignment"] = process_turns(
                    turns, audio, base / f"{condition}_features.jsonl",
                    batch_size=args.batch_size, max_utterances=args.max_utterances,
                    dictionary=args.mfa_dictionary, acoustic_model=args.mfa_acoustic_model,
                    conda_env=args.mfa_conda_env, mfa_config=args.mfa_config,
                    pitch_floor=args.pitch_floor, pitch_ceiling=args.pitch_ceiling,
                    metadata=metadata,
                )
        if args.validate_independent_asr:
            status["independent_asr_validation"] = validate_independent_asr(
                turns, audio, base / f"{condition}_independent_asr_validation.json",
                transcriber=transcriber, whisper_model=args.whisper_model,
                compute_type=args.compute_type, alignment_device=args.alignment_device,
                metadata=metadata,
            )
        if status:
            _write_json(base / f"{condition}_status.json", status)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
