"""Resumable, bounded feature extraction from locally staged speech datasets.

Use --dry-run to inspect a queue without importing acoustic/ASR dependencies.
HVC gets acoustic-only records; annotated English sources use MFA without ASR;
untranscribed narration uses short WhisperX clips followed by MFA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from acoustic_features import summarize_sound
from newman_ratner_pipeline import (
    ChatTurn, _audio_duration, _run, _write_json, chat_text_overlap,
    convert_chat, parse_chat, process_turns, process_whisperx_chat_turns,
    validate_independent_asr,
)


LOG = logging.getLogger("feature-batch")
SCHEMA_VERSION = "speech_features_v7"
DEFAULT_TIERS = {"acoustic_primary", "narration_supplement", "adult_supplement"}
SOURCE_NAMES = {"HVC", "AMI", "LibriVox", "NewmanRatner", "VoxPopuli",
                "YouTube_SciShow_pilot", "YouTube_selected_2000"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def _known_speaker(value: Any) -> str | None:
    """Normalize missing speaker IDs from manifests and dataset loaders."""
    if value is None:
        return None
    label = str(value).strip()
    return None if label.lower() in {"", "none", "null", "unknown", "nan"} else label


def _speaker_fields(row: dict, turn_speaker: str | None = None,
                    turn_attribution: str | None = None) -> dict:
    dataset = row["dataset"]
    listed = _known_speaker(row.get("speaker_id"))
    if dataset == "NewmanRatner":
        tier = _known_speaker(turn_speaker)
        speaker = (f"NewmanRatner:{row['age_group']}:{row['recording_id']}:{tier}"
                   if tier else None)
        return {"speaker_id": speaker, "speaker_label": tier,
                "speaker_role": "mother" if tier == "MOT" else None,
                "speaker_attribution": "CHAT_speaker_tier" if tier else "unresolved"}
    if dataset == "AMI":
        return {"speaker_id": listed, "speaker_label": listed,
                "speaker_role": "meeting_participant" if listed else None,
                "speaker_attribution": "headset_and_manual_word_annotation" if listed else "unresolved"}
    if dataset == "VoxPopuli":
        return {"speaker_id": listed, "speaker_label": listed,
                "speaker_role": "parliament_speaker" if listed else None,
                "speaker_attribution": "provided_segment_metadata" if listed else "unresolved"}
    if dataset == "HVC":
        return {"speaker_id": listed, "speaker_label": listed,
                "speaker_role": "adult_participant" if listed else None,
                "speaker_attribution": "paired_recording_manifest" if listed else "unresolved"}
    if dataset == "LibriVox":
        tier = _known_speaker(turn_speaker)
        if tier:
            return {"speaker_id": f"{row['source_id']}:{tier}", "speaker_label": tier,
                    "speaker_role": None,
                    "speaker_attribution": turn_attribution or "speaker_segments"}
        return {"speaker_id": listed, "speaker_label": listed,
                "speaker_role": "credited_narrator" if listed else None,
                "speaker_attribution": "catalogue_credit_unverified_per_utterance" if listed else "unresolved"}
    tier = _known_speaker(turn_speaker)
    if tier:
        return {"speaker_id": f"{row['source_id']}:{tier}", "speaker_label": tier,
                "speaker_role": None,
                "speaker_attribution": turn_attribution or "speaker_segments"}
    return {"speaker_id": None, "speaker_label": None,
            "speaker_role": None, "speaker_attribution": "unresolved"}


def _load_speaker_segments(path: Path) -> dict[str, list[dict]]:
    """Read optional, recording-timed speaker intervals for ASR sources."""
    by_source: dict[str, list[dict]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            try:
                source_id = str(item["source_id"])
                start, end = float(item["start_sec"]), float(item["end_sec"])
                speaker = _known_speaker(item["speaker_id"])
                attribution = str(item["attribution"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid speaker segment at {path}:{line_number}") from exc
            if (not source_id or not speaker or not math.isfinite(start)
                    or not math.isfinite(end) or not 0 <= start < end
                    or attribution not in {"manual_annotation", "external_diarization"}):
                raise ValueError(f"Invalid speaker segment at {path}:{line_number}")
            by_source.setdefault(source_id, []).append({
                "start_sec": start, "end_sec": end,
                "speaker_id": speaker, "attribution": attribution,
            })
    return by_source


def _speaker_for_interval(start: float, end: float, segments: list[dict]) -> tuple[str | None, str | None]:
    """Assign only a speaker covering at least 80% of an ASR utterance."""
    if end <= start or not segments:
        return None, None
    coverage: dict[str, float] = {}
    evidence: dict[str, set[str]] = {}
    for segment in segments:
        overlap = max(0.0, min(end, segment["end_sec"]) - max(start, segment["start_sec"]))
        if overlap:
            speaker = segment["speaker_id"]
            coverage[speaker] = coverage.get(speaker, 0.0) + overlap
            evidence.setdefault(speaker, set()).add(segment["attribution"])
    if not coverage:
        return None, None
    ranked = sorted(coverage.items(), key=lambda item: item[1], reverse=True)
    top, top_sec = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    duration = end - start
    if top_sec / duration < 0.8 or runner_up / duration > 0.1:
        return None, None
    sources = evidence[top]
    return top, "manual_annotation" if "manual_annotation" in sources else "external_diarization"


def _signature(audio: Path, annotation: Path | None = None) -> str:
    payload: list[Any] = []
    for path in (audio, annotation):
        if path is None:
            continue
        stat = path.stat()
        payload.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def _manifest_rows(path: Path, root: Path, selected: set[str]) -> list[dict]:
    rows = []
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["dataset"] not in selected or row["acquisition_status"] != "staged":
            continue
        if row["usage_tier"] not in DEFAULT_TIERS and row["dataset"] != "YouTube_SciShow_pilot":
            continue
        audio = root / row["relative_path"]
        if not audio.is_file():
            raise FileNotFoundError(audio)
        row["_audio"] = audio
        if row.get("annotation_path"):
            row["_annotation"] = root / row["annotation_path"]
        rows.append(row)
    return rows


def _newman_rows(root: Path) -> list[dict]:
    rows = []
    extracted = root / "transcripts" / "extracted"
    for condition in ("play", "interview"):
        audio_root = root / "media" / ("Interviews" if condition == "interview" else "")
        chat_root = extracted / ("Interviews" if condition == "interview" else "")
        suffix = ".mp3" if condition == "interview" else ".wav"
        for audio in sorted(audio_root.glob(f"*/*{suffix}")):
            age, stem = audio.parent.name, audio.stem
            chat = chat_root / age / f"{stem}.cha"
            if not chat.is_file():
                LOG.warning("No CHAT file for %s", audio)
                continue
            rows.append({
                "dataset": "NewmanRatner", "source_id": f"NewmanRatner:{age}:{stem}:{condition}",
                "recording_id": stem, "age_group": age, "source_section": condition,
                "audience_label": None, "language": "en", "speaker_id": "MOT",
                "transcript_status": "CHAT", "usage_tier": "annotation_review",
                "_audio": audio, "_annotation": chat,
            })
    return rows


def _vox_rows(path: Path, root: Path) -> list[dict]:
    """Read local VoxPopuli segment manifest; do not fetch a full release."""
    rows = []
    if not path.is_file():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("language", "en").lower() != "en":
            continue
        # Manifests staged on Windows must also resolve inside the Linux worker.
        audio = root / str(item["audio_path"]).replace("\\", "/")
        if not audio.is_file():
            raise FileNotFoundError(audio)
        transcript = (item.get("raw_text") or item.get("normalized_text") or "").strip()
        if not transcript:
            LOG.warning("No transcript for VoxPopuli %s", audio)
            continue
        rows.append({
            "dataset": "VoxPopuli", "source_id": item.get("source_id") or f"VoxPopuli:{item['audio_id']}",
            "recording_id": item["audio_id"], "audience_label": "general_audience",
            "audience_label_basis": "adult_parliamentary_context",
            "language": "en", "speaker_id": _known_speaker(item.get("speaker_id")),
            "split": item.get("split"), "is_gold_transcript": item.get("is_gold_transcript"),
            "sha256": item.get("sha256"),
            "format": "parliamentary_speech", "speaker_type": "adult",
            "license": "CC0", "attribution": "VoxPopuli / European Parliament",
            "transcript_status": "provided_segment_text", "usage_tier": "adult_supplement",
            "_audio": audio, "_transcript": transcript,
        })
    return rows


def _selected_youtube_rows(manifest: Path, audio_root: Path) -> list[dict]:
    """Include only selected-video audio that is already complete on disk."""
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        candidates = list(csv.DictReader(stream))
    by_id = {}
    for item in candidates:
        video_id = item["video_id"]
        if video_id not in by_id or item.get("yt_dlp_ok") == "True":
            by_id[video_id] = item
    rows = []
    extensions = {".m4a", ".webm", ".opus", ".mp3", ".ogg"}
    for video_id, item in sorted(by_id.items()):
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            continue
        files = [path for path in audio_root.glob(f"{video_id}.*")
                 if path.suffix.lower() in extensions and path.stat().st_size > 0]
        if len(files) != 1:
            if len(files) > 1:
                LOG.warning("Multiple audio files for selected YouTube ID %s", video_id)
            continue
        rows.append({
            "dataset": "YouTube_selected_2000", "source_id": f"YouTube:{video_id}",
            "recording_id": video_id, "audience_label": None,
            "audience_label_basis": "selection_candidate_only",
            "candidate_kind": item.get("kind"), "title": item.get("title") or item.get("title_discovered"),
            "channel": item.get("channel_discovered"), "language": None,
            "speaker_id": None, "transcript_status": "whisperx_required",
            "usage_tier": "exploratory_hold", "_audio": files[0],
        })
    return rows


def build_queue(args: argparse.Namespace) -> list[dict]:
    selected = set(args.sources.split(","))
    unknown = selected - SOURCE_NAMES
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")
    staged_sources = selected - {"NewmanRatner", "VoxPopuli", "YouTube_selected_2000"}
    queue = _manifest_rows(args.manifest, args.root, staged_sources) if staged_sources else []
    if "NewmanRatner" in selected:
        queue.extend(_newman_rows(args.newman_root))
    if "VoxPopuli" in selected:
        if args.vox_manifest is None:
            raise ValueError("VoxPopuli requires --vox-manifest with local audio/text rows")
        queue.extend(_vox_rows(args.vox_manifest, args.root))
    if "YouTube_selected_2000" in selected:
        queue.extend(_selected_youtube_rows(args.youtube_manifest, args.youtube_root))
    identifiers = [row["source_id"] for row in queue]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Duplicate source_id in feature queue")
    return sorted(queue, key=lambda row: (row["dataset"], row["source_id"]))


def _public_metadata(row: dict) -> dict:
    return {
        key: value for key, value in row.items()
        if not key.startswith("_") and key not in {"download_url", "source_page_url", "notes"}
    }


def _base_record(row: dict, utterance_id: str) -> dict:
    audio = row["_audio"]
    return {
        "schema_version": SCHEMA_VERSION,
        "record_id": f"{row['source_id']}:{utterance_id}",
        "source_signature": row["_record_signature"],
        "dataset": row["dataset"], "source_id": row["source_id"],
        "recording_id": row["recording_id"],
        "utterance_id": utterance_id,
        "audio_path": str(audio.resolve()),
        "language": row.get("language") or row.get("_detected_language"),
        **_speaker_fields(row),
        "audience_label": row.get("audience_label"),
        "audience_label_basis": row.get("audience_label_basis") or row.get("usage_tier"),
        "extraction_settings": row["_extraction_settings"],
        "software_versions": row["_software_versions"],
        "source_metadata": _public_metadata(row),
    }


def _verify_audio_checksum(row: dict) -> None:
    expected = str(row.get("sha256") or "").lower()
    if not expected:
        return
    digest = hashlib.sha256()
    with row["_audio"].open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"Audio SHA-256 mismatch for {row['source_id']}")


class FinalWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen: dict[str, str] = {}
        self.source_signatures: dict[str, set[str]] = {}
        if path.is_file():
            with path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        key = row["record_id"]
                        signature = row["source_signature"]
                        source_id = row["source_id"]
                    except (ValueError, KeyError) as exc:
                        raise ValueError(f"Invalid final JSONL at {path}:{line_number}") from exc
                    if key in self.seen:
                        raise ValueError(f"Duplicate final record: {key}")
                    self.seen[key] = signature
                    self.source_signatures.setdefault(source_id, set()).add(signature)

    def append(self, records: Iterable[dict]) -> int:
        added = 0
        with self.path.open("a", encoding="utf-8") as stream:
            for record in records:
                key, signature = record["record_id"], record["source_signature"]
                if key in self.seen:
                    if self.seen[key] != signature:
                        raise ValueError(f"Source changed for final record {key}; choose a new output directory")
                    continue
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self.seen[key] = signature
                self.source_signatures.setdefault(record["source_id"], set()).add(signature)
                added += 1
            stream.flush()
            os.fsync(stream.fileno())
        return added


def _audio_chunks(duration: float, max_clip_sec: float) -> Iterable[tuple[int, float, float]]:
    for index in range(math.ceil(duration / max_clip_sec)):
        start = index * max_clip_sec
        end = min(duration, (index + 1) * max_clip_sec)
        if end - start > 0.05:
            yield index, start, end


def _extract_clip(audio: Path, start: float, end: float, output: Path, sample_rate: int = 16000) -> None:
    _run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(audio),
          "-t", str(end - start), "-ac", "1", "-ar", str(sample_rate), "-vn", str(output)])


def _acoustic_summary(clip: Path, pitch_floor: float, pitch_ceiling: float) -> dict:
    import parselmouth

    sound = parselmouth.Sound(str(clip))
    if sound.n_channels > 1:
        sound = sound.convert_to_mono()
    pitch = sound.to_pitch_ac(pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    return summarize_sound(sound, pitch)


def _hvc_records(row: dict, args: argparse.Namespace) -> Iterable[dict]:
    audio = row["_audio"]
    duration = _audio_duration(audio)
    with tempfile.TemporaryDirectory(prefix="hvc_clips_", dir=args.out_dir) as temporary:
        for index, start, end in _audio_chunks(duration, args.max_clip_sec):
            clip = Path(temporary) / f"seg_{index:05d}.wav"
            _extract_clip(audio, start, end, clip)
            record = _base_record(row, f"seg_{index:05d}")
            record.update({
                "start_sec": round(start, 3), "end_sec": round(end, 3),
                "transcript": None, "transcript_source": "none",
                "alignment_method": None, "words": [], "syllables": [], "phones": [],
                "phone_set": None, "alignment_qc": None,
                "annotations": {}, "dependent_tiers": {},
                "acoustic_features": _acoustic_summary(clip, args.pitch_floor, args.pitch_ceiling),
                "quality_flags": ["no_lexical_transcript", "intensity_not_level_calibrated"]
                + (["speaker_unresolved"] if record["speaker_id"] is None else []),
            })
            yield record


def _ami_words(path: Path) -> list[dict]:
    words = []
    for item in ET.parse(path).getroot().iter():
        if item.tag != "w" or item.get("punc") == "true":
            continue
        token = (item.text or "").strip()
        if not token or not re.search(r"[A-Za-z]", token):
            continue
        start, end = float(item.get("starttime", "nan")), float(item.get("endtime", "nan"))
        if math.isfinite(start) and math.isfinite(end) and 0 <= start < end:
            words.append({"word": token, "start": start, "end": end})
    return sorted(words, key=lambda item: (item["start"], item["end"]))


def _ami_turns(row: dict, max_clip_sec: float) -> list[ChatTurn]:
    annotation = row.get("_annotation")
    if annotation is None or not annotation.is_file():
        raise FileNotFoundError(annotation)
    words = _ami_words(annotation)
    groups: list[list[dict]] = []
    group: list[dict] = []
    for word in words:
        if word["end"] - word["start"] > max_clip_sec:
            LOG.warning("Skipping AMI word longer than clip limit: %s", word)
            continue
        if group and (word["start"] - group[-1]["end"] > 1.5
                      or word["end"] - group[0]["start"] > max_clip_sec
                      or len(group) >= 80):
            groups.append(group)
            group = []
        group.append(word)
    if group:
        groups.append(group)
    return [ChatTurn(
        f"utt_{index:05d}", row.get("speaker_id") or "UNKNOWN",
        group[0]["start"], group[-1]["end"],
        " ".join(word["word"] for word in group),
        " ".join(word["word"] for word in group),
        annotations={"manual_words": group},
    ) for index, group in enumerate(groups)]


def _newman_turns(row: dict, args: argparse.Namespace) -> tuple[list[ChatTurn], dict]:
    chat = row["_annotation"]
    age, stem = row["age_group"], row["recording_id"]
    extracted = args.newman_root / "transcripts" / "extracted"
    main_chat = extracted / age / f"{stem}.cha"
    interview_chat = extracted / "Interviews" / age / f"{stem}.cha"
    overlap = (chat_text_overlap(main_chat, interview_chat)
               if main_chat.is_file() and interview_chat.is_file() else None)
    headers, all_turns = parse_chat(chat)
    turns, counts = convert_chat(chat, max_duration_sec=args.max_clip_sec)
    row["_conversion_counts"] = counts
    transcript_file = args.out_dir / "transcripts" / age / stem / f"{row['source_section']}.json"
    _write_json(transcript_file, {
        "schema_version": "newman_ratner_transcript_v1", "source_chat": str(chat.resolve()),
        "source_audio": str(row["_audio"].resolve()), "headers": headers,
        "all_turns": all_turns, "selected_utterances": [asdict(turn) for turn in turns],
        "conversion_counts": counts, "pair_chat_text_overlap": overlap,
    })
    status = "near_duplicate_CHAT_text_do_not_compare" if overlap is not None and overlap >= 0.8 else "requires_audio_review"
    return turns, {"pair_chat_text_overlap": overlap, "paired_contrast_status": status,
                   "source_chat": str(chat.resolve())}


def _asr_turns(row: dict, args: argparse.Namespace) -> list[ChatTurn]:
    """Transcribe one bounded clip at a time and resume completed ASR chunks."""
    from speech_pipeline import GPUTranscriber

    audio = row["_audio"]
    cache_dir = args.out_dir / "asr" / _safe_name(row["source_id"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    config_path = cache_dir / "config.json"
    config = {"source_signature": _signature(audio), "model": args.whisper_model,
              "language": row.get("language"), "max_clip_sec": args.max_clip_sec}
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"ASR input/settings changed for {row['source_id']}")
    else:
        if list(cache_dir.glob("chunk_*.json")):
            raise ValueError(f"ASR cache has no matching config: {cache_dir}")
        _write_json(config_path, config)
    transcriber = None
    turns = []
    duration = _audio_duration(audio)
    for chunk_index, start, end in _audio_chunks(duration, args.max_clip_sec):
        cache = cache_dir / f"chunk_{chunk_index:05d}.json"
        if cache.is_file():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            segments = cached["segments"]
            language = cached.get("language") or row.get("language")
        else:
            if transcriber is None:
                transcriber = GPUTranscriber(args.asr_device, model_size=args.whisper_model,
                                             language=row.get("language"), batch_size=1)
            with tempfile.TemporaryDirectory(prefix="asr_clip_", dir=cache_dir) as temporary:
                clip = Path(temporary) / "clip.wav"
                _extract_clip(audio, start, end, clip)
                try:
                    segments, language = transcriber.transcribe_segments(clip)
                except ValueError as exc:
                    if "no timed speech segments" not in str(exc):
                        raise
                    segments, language = [], row.get("language")
                if segments and language != "en":
                    raise ValueError(f"ASR detected {language!r} for {row['source_id']}")
            _write_json(cache, {"start_sec": start, "end_sec": end,
                                "language": language, "segments": segments})
        if segments and language != "en":
            raise ValueError(f"Cached ASR language {language!r} is unsupported for {row['source_id']}")
        if language == "en":
            row["_detected_language"] = "en"
        for segment_index, segment in enumerate(segments):
            text = re.sub(r"\s+", " ", segment["text"]).strip()
            absolute_start = start + segment["start"]
            absolute_end = min(end, start + segment["end"])
            if text and 0 <= absolute_start < absolute_end <= duration + 0.1:
                speaker, attribution = _speaker_for_interval(
                    absolute_start, absolute_end, row.get("_speaker_segments", [])
                )
                turns.append(ChatTurn(
                    f"chunk_{chunk_index:05d}_utt_{segment_index:03d}",
                    speaker or "UNKNOWN", absolute_start,
                    absolute_end, text, text,
                    annotations={"asr_chunk_index": chunk_index,
                                 "chunk_boundary_risk": True,
                                 "speaker_attribution": attribution},
                ))
    return turns


def _transcribed_records(row: dict, args: argparse.Namespace) -> Iterable[dict]:
    dataset = row["dataset"]
    metadata = {"dataset": dataset, "source_id": row["source_id"]}
    newman_backend = getattr(args, "newman_alignment_backend", "whisperx_chat")
    if dataset == "NewmanRatner":
        turns, extra = _newman_turns(row, args)
        metadata.update(extra)
        transcript_source = "CHAT"
    elif dataset == "AMI":
        turns = _ami_turns(row, args.max_clip_sec)
        transcript_source = "AMI_manual_word_xml"
    elif dataset == "VoxPopuli":
        duration = _audio_duration(row["_audio"])
        if duration > args.max_clip_sec:
            raise ValueError(f"VoxPopuli segment exceeds {args.max_clip_sec}s: {row['source_id']}")
        text = row["_transcript"]
        turns = [ChatTurn("utt_00000", row.get("speaker_id") or "UNKNOWN", 0.0,
                          duration, text, text)]
        transcript_source = "VoxPopuli_provided"
    else:
        turns = _asr_turns(row, args)
        metadata["asr_model"] = args.whisper_model
        transcript_source = "whisperx_asr"
    work = args.out_dir / "per_source" / _safe_name(row["source_id"])
    work.mkdir(parents=True, exist_ok=True)
    if not turns:
        row["_alignment_result"] = {"eligible": 0, "completed_now": 0}
        _write_json(work / "status.json", {"eligible": 0, "completed_now": 0,
                                             "reason": "no_eligible_timed_utterances"})
        return
    if dataset == "NewmanRatner" and newman_backend == "whisperx_chat":
        transcriber = getattr(args, "_newman_transcriber", None)
        if transcriber is None:
            from speech_pipeline import GPUTranscriber

            transcriber = GPUTranscriber(
                args.asr_device, model_size=args.whisper_model,
                compute_type=getattr(args, "compute_type", "float16"), language="en",
                alignment_device=getattr(args, "alignment_device", None),
            )
            args._newman_transcriber = transcriber
        result = process_whisperx_chat_turns(
            turns, row["_audio"], work / "features.jsonl", transcriber=transcriber,
            batch_size=args.batch_size, max_utterances=args.limit_utterances,
            pitch_floor=args.pitch_floor, pitch_ceiling=args.pitch_ceiling,
            chat_context_sec=getattr(args, "newman_chat_context_sec", 0.25),
            whisper_model=args.whisper_model,
            compute_type=getattr(args, "compute_type", "float16"),
            alignment_device=getattr(args, "alignment_device", None), metadata=metadata,
        )
        if getattr(args, "newman_validate_independent_asr", False):
            result["independent_asr_validation"] = validate_independent_asr(
                turns, row["_audio"], work / "independent_asr_validation.json",
                transcriber=transcriber, whisper_model=args.whisper_model,
                compute_type=getattr(args, "compute_type", "float16"),
                alignment_device=getattr(args, "alignment_device", None), metadata=metadata,
            )
    else:
        result = process_turns(
            turns, row["_audio"], work / "features.jsonl",
            batch_size=args.batch_size, max_utterances=args.limit_utterances,
            dictionary=args.mfa_dictionary,
            acoustic_model=args.mfa_acoustic_model, conda_env=args.mfa_conda_env,
            pitch_floor=args.pitch_floor, pitch_ceiling=args.pitch_ceiling,
            metadata=metadata, transcript_source=transcript_source,
        )
    row["_alignment_result"] = result
    _write_json(work / "status.json", result)
    if result["out_of_audio_bounds"]:
        LOG.warning("%s: %d turns outside audio", row["source_id"], result["out_of_audio_bounds"])
    with (work / "features.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            record = _base_record(row, item["utterance_id"])
            record.update(_speaker_fields(
                row, item.get("speaker"),
                item.get("annotations", {}).get("speaker_attribution"),
            ))
            source_flags = []
            if dataset == "LibriVox" and record["speaker_attribution"] == "catalogue_credit_unverified_per_utterance":
                source_flags.extend(["narration_intro_outro_not_reviewed",
                                     "catalogue_speaker_not_verified_per_utterance"])
            elif dataset in {"YouTube_SciShow_pilot", "YouTube_selected_2000"}:
                source_flags.append("exploratory_audience_proxy")
            elif dataset in {"AMI", "VoxPopuli"}:
                source_flags.append("adult_genre_supplement")
            if item.get("annotations", {}).get("chunk_boundary_risk"):
                source_flags.append("asr_chunk_boundary_risk")
            if record["speaker_id"] is None:
                source_flags.append("speaker_unresolved")
                if row.get("_speaker_segments"):
                    source_flags.append("speaker_segment_ambiguous_or_uncovered")
            record.update({
                "start_sec": item["start_sec"], "end_sec": item["end_sec"],
                "transcript": item["transcript"],
                "transcript_source": item["transcript_source"],
                "alignment_method": item["alignment_method"],
                "alignment_scope": item.get("alignment_scope", "word_and_phone"),
                "phone_set": item["phone_set"],
                "words": item["words"], "syllables": item["syllables"],
                "phones": item["phones"],
                "chat_time_link_sec": item.get("chat_time_link_sec"),
                "alignment_search_window_sec": item.get("alignment_search_window_sec"),
                "alignment_qc": item.get("alignment_qc", {}),
                "acoustic_features": item.get("acoustic_features"),
                "annotations": item.get("annotations", {}),
                "dependent_tiers": item.get("dependent_tiers", {}),
                "quality_flags": item.get("alignment_qc", {}).get("quality_flags", []) + source_flags + (
                    ["audience_label_unverified", "chat_pair_text_overlap"]
                    if dataset == "NewmanRatner" and
                    metadata.get("paired_contrast_status") == "near_duplicate_CHAT_text_do_not_compare"
                    else (["audience_label_unverified"] if dataset == "NewmanRatner" else [])
                ),
            })
            if dataset == "NewmanRatner":
                record["audience_label"] = None
                record["source_metadata"].update(extra)
            yield record


def _dependency_check(queue: list[dict], args: argparse.Namespace) -> None:
    import importlib.util

    for module in ("numpy", "parselmouth"):
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"Missing {module}; run in the speech-features environment")
    newman_backend = getattr(args, "newman_alignment_backend", "whisperx_chat")
    mfa_required = any(
        row["dataset"] != "HVC" and not (
            row["dataset"] == "NewmanRatner" and newman_backend == "whisperx_chat"
        )
        for row in queue
    )
    if mfa_required:
        from mfa_alignment import MFAAligner
        MFAAligner(args.mfa_dictionary, args.mfa_acoustic_model,
                   conda_env=args.mfa_conda_env).ensure_available()
    whisperx_required = any(
        row["dataset"] in {"LibriVox", "YouTube_SciShow_pilot", "YouTube_selected_2000"}
        or (row["dataset"] == "NewmanRatner" and newman_backend == "whisperx_chat")
        for row in queue
    )
    if whisperx_required:
        if importlib.util.find_spec("whisperx") is None:
            raise RuntimeError("WhisperX is required for sources without transcripts")
        if args.asr_device.startswith("cuda"):
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("ASR device is CUDA, but no GPU is available; use --asr-device cpu or a GPU host")


def _record_signature(row: dict, args: argparse.Namespace) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "input_signature": _signature(row["_audio"], row.get("_annotation")),
        "transcript": row.get("_transcript"),
        "source_metadata": _public_metadata(row),
        "max_clip_sec": args.max_clip_sec,
        "pitch_floor": args.pitch_floor, "pitch_ceiling": args.pitch_ceiling,
        "mfa_dictionary": args.mfa_dictionary,
        "mfa_acoustic_model": args.mfa_acoustic_model,
        "mfa_conda_env": getattr(args, "mfa_conda_env", None),
        "asr_device": getattr(args, "asr_device", None) if row["dataset"] in
        {"LibriVox", "YouTube_SciShow_pilot", "YouTube_selected_2000", "NewmanRatner"} else None,
        "alignment_device": getattr(args, "alignment_device", None)
        if row["dataset"] == "NewmanRatner" else None,
        "compute_type": getattr(args, "compute_type", None)
        if row["dataset"] == "NewmanRatner" else None,
        "speaker_segments_signature": row.get("_speaker_segments_signature"),
        "software_versions": row.get("_software_versions"),
        "whisper_model": args.whisper_model if row["dataset"] in
        {"LibriVox", "YouTube_SciShow_pilot", "YouTube_selected_2000", "NewmanRatner"} else None,
        "newman_alignment_backend": getattr(args, "newman_alignment_backend", None)
        if row["dataset"] == "NewmanRatner" else None,
        "newman_chat_context_sec": getattr(args, "newman_chat_context_sec", None)
        if row["dataset"] == "NewmanRatner" else None,
        "newman_validate_independent_asr": getattr(args, "newman_validate_independent_asr", None)
        if row["dataset"] == "NewmanRatner" else None,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _software_versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    names = {
        "numpy": "numpy", "parselmouth": "praat-parselmouth",
        "mfa": "montreal-forced-aligner", "whisperx": "whisperx",
    }
    result = {}
    for label, package in names.items():
        try:
            result[label] = version(package)
        except PackageNotFoundError:
            result[label] = None
    return result


def run(args: argparse.Namespace) -> dict:
    queue = build_queue(args)
    speaker_segments_path = getattr(args, "speaker_segments", None)
    if speaker_segments_path is not None:
        by_source = _load_speaker_segments(speaker_segments_path)
        allowed = {row["source_id"] for row in queue
                   if row["dataset"] in {"LibriVox", "YouTube_SciShow_pilot",
                                         "YouTube_selected_2000"}}
        unknown = set(by_source) - allowed
        if unknown:
            raise ValueError(f"Speaker segments refer to unavailable ASR sources: {sorted(unknown)}")
        digest = hashlib.sha256(speaker_segments_path.read_bytes()).hexdigest()
        for row in queue:
            row["_speaker_segments"] = by_source.get(row["source_id"], [])
            row["_speaker_segments_signature"] = digest if row["source_id"] in by_source else None
    if args.source_id:
        requested = set(args.source_id)
        available = {row["source_id"] for row in queue}
        missing = requested - available
        if missing:
            raise ValueError(f"Source IDs not found in the selected queue: {sorted(missing)}")
        queue = [row for row in queue if row["source_id"] in requested]
    if args.limit_recordings:
        queue = queue[:args.limit_recordings]
    counts = {name: sum(row["dataset"] == name for row in queue) for name in sorted({r["dataset"] for r in queue})}
    if args.dry_run:
        return {"mode": "dry_run", "queued_recordings": len(queue), "by_dataset": counts}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lock = args.out_dir / ".feature_run.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f"Another run may be active ({lock}). Check it before removing the lock.") from exc
    try:
        _dependency_check(queue, args)
        versions = _software_versions()
        for row in queue:
            used_versions = {key: value for key, value in versions.items()
                             if key in {"numpy", "parselmouth"} or
                             (key == "mfa" and row["dataset"] != "HVC" and not (
                                 row["dataset"] == "NewmanRatner" and
                                 getattr(args, "newman_alignment_backend", "whisperx_chat") == "whisperx_chat"
                             )) or
                             (key == "whisperx" and row["dataset"] in
                              {"LibriVox", "YouTube_SciShow_pilot", "YouTube_selected_2000", "NewmanRatner"})}
            row["_software_versions"] = used_versions
            row["_extraction_settings"] = {
                "max_clip_sec": args.max_clip_sec,
                "pitch_floor_hz": args.pitch_floor,
                "pitch_ceiling_hz": args.pitch_ceiling,
                "mfa_dictionary": args.mfa_dictionary if row["dataset"] != "HVC" and not (
                    row["dataset"] == "NewmanRatner" and
                    getattr(args, "newman_alignment_backend", "whisperx_chat") == "whisperx_chat"
                ) else None,
                "mfa_acoustic_model": args.mfa_acoustic_model if row["dataset"] != "HVC" and not (
                    row["dataset"] == "NewmanRatner" and
                    getattr(args, "newman_alignment_backend", "whisperx_chat") == "whisperx_chat"
                ) else None,
                "whisper_model": args.whisper_model if row["dataset"] in
                {"LibriVox", "YouTube_SciShow_pilot", "YouTube_selected_2000", "NewmanRatner"} else None,
                "newman_alignment_backend": getattr(args, "newman_alignment_backend", None)
                if row["dataset"] == "NewmanRatner" else None,
                "newman_chat_context_sec": getattr(args, "newman_chat_context_sec", None)
                if row["dataset"] == "NewmanRatner" else None,
                "newman_validate_independent_asr": getattr(args, "newman_validate_independent_asr", None)
                if row["dataset"] == "NewmanRatner" else None,
            }
            row["_record_signature"] = _record_signature(row, args)
        writer = FinalWriter(args.out_dir / "features.jsonl")
        status_path = args.out_dir / "run_status.json"
        statuses = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        for index, row in enumerate(queue, 1):
            key = row["source_id"]
            previous = statuses.get(key, {})
            if (previous.get("status") == "completed" and
                    previous.get("source_signature") == row["_record_signature"] and
                    writer.source_signatures.get(key) == {row["_record_signature"]}):
                LOG.info("%d/%d %s: already completed", index, len(queue), key)
                continue
            try:
                _verify_audio_checksum(row)
                records = _hvc_records(row, args) if row["dataset"] == "HVC" else _transcribed_records(row, args)
                added = writer.append(records)
                partial = (row["dataset"] != "HVC" and args.limit_utterances > 0)
                no_eligible = row.get("_alignment_result", {}).get("eligible") == 0
                status = "no_eligible" if no_eligible else ("partial_limit" if partial else "completed")
                statuses[key] = {
                    "status": status, "new_records": added,
                    "source_signature": row["_record_signature"],
                    "conversion_counts": row.get("_conversion_counts"),
                    "alignment_result": row.get("_alignment_result"),
                }
                LOG.info("%d/%d %s: %d new records", index, len(queue), key, added)
            except Exception as exc:
                statuses[key] = {"status": "failed", "error": str(exc)[-1000:]}
                LOG.exception("%d/%d %s failed", index, len(queue), key)
                _write_json(status_path, statuses)
                if args.stop_on_error:
                    raise
                continue
            _write_json(status_path, statuses)
        return {"mode": "run", "queued_recordings": len(queue), "by_dataset": counts,
                "completed": sum(statuses.get(r["source_id"], {}).get("status") == "completed" for r in queue),
                "partial_limit": sum(statuses.get(r["source_id"], {}).get("status") == "partial_limit" for r in queue),
                "no_eligible": sum(statuses.get(r["source_id"], {}).get("status") == "no_eligible" for r in queue),
                "failed": sum(statuses.get(r["source_id"], {}).get("status") == "failed" for r in queue),
                "jsonl": str(writer.path.resolve())}
    finally:
        lock.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/weekend_staged_manifest.jsonl"))
    parser.add_argument("--newman-root", type=Path, default=Path("data/staging/newman_ratner"))
    parser.add_argument("--vox-manifest", type=Path)
    parser.add_argument("--youtube-manifest", type=Path,
                        default=Path("artifacts/videos_selected_all_2000.csv"))
    parser.add_argument("--youtube-root", type=Path,
                        default=Path("data/staging/youtube_selected_2000/audio"))
    parser.add_argument("--speaker-segments", type=Path,
                        help="Optional JSONL of recording-timed speaker intervals for ASR sources")
    parser.add_argument("--out-dir", type=Path, default=Path("results/full_features"))
    parser.add_argument("--sources", default="HVC,AMI,LibriVox,NewmanRatner",
                        help="Comma-separated source names")
    parser.add_argument("--max-clip-sec", type=float, default=25.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-recordings", type=int, default=0)
    parser.add_argument("--source-id", action="append", default=[],
                        help="Process an exact source ID; may be repeated")
    parser.add_argument("--limit-utterances", type=int, default=0,
                        help="Pilot limit per transcribed recording; 0 means all")
    parser.add_argument("--pitch-floor", type=float, default=75.0)
    parser.add_argument("--pitch-ceiling", type=float, default=600.0)
    parser.add_argument("--mfa-dictionary", default="english_us_mfa")
    parser.add_argument("--mfa-acoustic-model", default="english_mfa")
    parser.add_argument("--mfa-conda-env", default=None)
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--alignment-device", default=None)
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--newman-alignment-backend", choices=("whisperx_chat", "mfa"),
                        default="whisperx_chat")
    parser.add_argument("--newman-chat-context-sec", type=float, default=0.25)
    parser.add_argument("--newman-validate-independent-asr", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or not math.isfinite(args.max_clip_sec) or
            args.max_clip_sec <= 0 or args.limit_recordings < 0
            or args.limit_utterances < 0 or args.newman_chat_context_sec < 0):
        parser.error("batch size and clip seconds must be positive; limit cannot be negative")
    if args.batch_size > 8 or args.max_clip_sec > 30:
        parser.error("bounded processing requires --batch-size <= 8 and --max-clip-sec <= 30")
    if not 0 < args.pitch_floor < args.pitch_ceiling:
        parser.error("invalid pitch range")
    result = run(args)
    print(json.dumps(result, indent=2))
    return 1 if result.get("failed", 0) else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
