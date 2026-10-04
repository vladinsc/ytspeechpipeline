"""Small Newman/Ratner GPU pilot for speaker, ASR, and word-alignment review.

Speaker clusters and ASR text come from audio. CHAT is used only as the
reference for evaluation and as the transcript for the separate guided
alignment track. Anonymous diarization clusters are mapped to CHAT tiers only
when calculating the report; that mapping is not fed into either model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from newman_ratner_pipeline import (
    ChatTurn, _audio_duration, _word_error_rate, _write_json,
    clean_chat_text, parse_chat, process_whisperx_chat_turns,
    validate_independent_asr,
)
from speech_pipeline import GPUTranscriber


LOG = logging.getLogger("newman-quality-pilot")
DEFAULT_SOURCES = (
    "NewmanRatner:07:4269LP:play",
    "NewmanRatner:18:5694MC:interview",
    "NewmanRatner:24:7120CB:interview",
)
TIERS = {"MOT", "EXP", "CHI"}


def source_paths(root: Path, source_id: str) -> tuple[Path, Path]:
    match = re.fullmatch(
        r"NewmanRatner:(07|10|11|18|24):([A-Za-z0-9_-]+):(play|interview)",
        source_id,
    )
    if match is None:
        raise ValueError(f"Invalid Newman/Ratner source ID: {source_id}")
    age, stem, condition = match.groups()
    section = ("Interviews",) if condition == "interview" else ()
    audio = root / "media" / Path(*section) / age / (stem + (".mp3" if section else ".wav"))
    chat = root / "transcripts" / "extracted" / Path(*section) / age / (stem + ".cha")
    if not audio.is_file() or not chat.is_file():
        raise FileNotFoundError(f"Pilot source needs audio and CHAT: {audio}; {chat}")
    return audio, chat


def all_timed_turns(chat: Path, audio_duration: float) -> list[ChatTurn]:
    """Keep all intelligible, single-link adult and child turns for evaluation."""
    turns = []
    for row in parse_chat(chat)[1]:
        if row["speaker"] not in TIERS or len(row["time_links_ms"]) != 1:
            continue
        start_ms, end_ms = row["time_links_ms"][0]
        text = clean_chat_text(row["chat_text"])
        if (text is None or not 0 <= start_ms < end_ms
                or (end_ms - start_ms) > 25_000
                or len(text.split()) / ((end_ms - start_ms) / 1000) > 8.0
                or end_ms / 1000 > audio_duration + 0.05):
            continue
        turns.append(ChatTurn(
            row["utterance_id"], row["speaker"], start_ms / 1000,
            end_ms / 1000, text, row["chat_text"], row["dependent_tiers"],
        ))
    return sorted(turns, key=lambda item: (item.start_sec, item.utterance_id))


def sample_turns(turns: list[ChatTurn], per_speaker: int) -> list[ChatTurn]:
    """Choose turns across each speaker's full recording timeline."""
    selected = []
    for speaker in sorted(TIERS):
        pool = [turn for turn in turns if turn.speaker == speaker]
        if len(pool) <= per_speaker:
            selected.extend(pool)
        elif per_speaker == 1:
            selected.append(pool[len(pool) // 2])
        else:
            indices = {round(index * (len(pool) - 1) / (per_speaker - 1))
                       for index in range(per_speaker)}
            selected.extend(pool[index] for index in sorted(indices))
    return sorted(selected, key=lambda item: (item.start_sec, item.utterance_id))


def _best_mapping(overlap_seconds: dict[tuple[str, str], float]) -> dict[str, str]:
    """Map anonymous clusters to CHAT tiers for scoring, without model leakage."""
    predicted = sorted({pred for pred, _ref in overlap_seconds})
    reference = sorted({ref for _pred, ref in overlap_seconds})
    if not predicted or not reference:
        return {}
    best_score = -1.0
    best = {}
    if len(predicted) >= len(reference):
        choices = (dict(zip(chosen, reference))
                   for chosen in itertools.permutations(predicted, len(reference)))
    else:
        choices = (dict(zip(predicted, chosen))
                   for chosen in itertools.permutations(reference, len(predicted)))
    for mapping in choices:
        score = sum(overlap_seconds.get((pred, ref), 0.0)
                    for pred, ref in mapping.items())
        if score > best_score:
            best_score, best = score, mapping
    return best


def evaluate_speakers(turns: list[ChatTurn], diarization: list[dict]) -> dict:
    """Score only spans where CHAT has one speaker; keep missed speech visible."""
    reference = [turn for turn in turns if turn.end_sec - turn.start_sec >= 0.3]
    points = sorted({time for turn in reference for time in (turn.start_sec, turn.end_sec)}
                    | {time for row in diarization for time in (row["start"], row["end"])})
    overlap: dict[tuple[str, str], float] = defaultdict(float)
    reference_seconds = 0.0
    covered_seconds = 0.0
    missed_seconds = 0.0
    predicted_overlap_seconds = 0.0
    for start, end in zip(points, points[1:]):
        midpoint = (start + end) / 2
        refs = {turn.speaker for turn in reference
                if turn.start_sec <= midpoint < turn.end_sec}
        if len(refs) != 1:
            continue
        duration = end - start
        reference_seconds += duration
        preds = {row["speaker"] for row in diarization
                 if row["start"] <= midpoint < row["end"]}
        if len(preds) == 1:
            overlap[(next(iter(preds)), next(iter(refs)))] += duration
            covered_seconds += duration
        elif not preds:
            missed_seconds += duration
        else:
            predicted_overlap_seconds += duration
    mapping = _best_mapping(overlap)
    correct = sum(value for (pred, ref), value in overlap.items()
                  if mapping.get(pred) == ref)
    return {
        "reference": "exclusive_timed_CHAT_tier_spans_at_least_0.3_sec",
        "cluster_to_CHAT_mapping_for_scoring_only": mapping,
        "reference_seconds": round(reference_seconds, 3),
        "covered_seconds": round(covered_seconds, 3),
        "missed_seconds": round(missed_seconds, 3),
        "predicted_overlap_seconds": round(predicted_overlap_seconds, 3),
        "speaker_coverage": round(covered_seconds / reference_seconds, 3)
        if reference_seconds else None,
        "mapped_accuracy_on_covered_speech": round(correct / covered_seconds, 3)
        if covered_seconds else None,
        "mapped_accuracy_including_misses": round(correct / reference_seconds, 3)
        if reference_seconds else None,
        "overlap_seconds_by_cluster_and_CHAT_tier": [
            {"cluster": pred, "chat_tier": ref, "seconds": round(seconds, 3)}
            for (pred, ref), seconds in sorted(overlap.items())
        ],
    }


def evaluate_asr(turns: list[ChatTurn], asr_words: list[dict]) -> dict:
    """Compare independent ASR words inside exclusive CHAT turn windows."""
    eligible = [turn for turn in turns if not any(
        other.utterance_id != turn.utterance_id
        and other.speaker != turn.speaker
        and min(turn.end_sec, other.end_sec) - max(turn.start_sec, other.start_sec) > 0.05
        for other in turns
    )]
    errors = tokens = 0
    checks = []
    for turn in eligible:
        hypothesis = " ".join(str(word["word"]) for word in asr_words
                              if "start" in word and "end" in word
                              and turn.start_sec <= (word["start"] + word["end"]) / 2 < turn.end_sec)
        comparison = _word_error_rate(turn.transcript, hypothesis)
        errors += comparison["edit_distance"]
        tokens += comparison["reference_token_count"]
        checks.append({
            "utterance_id": turn.utterance_id, "speaker": turn.speaker,
            "start_sec": turn.start_sec, "end_sec": turn.end_sec,
            "chat_text": turn.transcript, "asr_text": hypothesis,
            **comparison,
        })
    return {
        "method": "ASR_word_midpoints_inside_nonoverlapping_CHAT_turns",
        "diagnostic_wer": round(errors / tokens, 3) if tokens else None,
        "reference_token_count": tokens, "edit_distance": errors,
        "turns_scored": len(checks),
        "turns_excluded_for_cross_speaker_overlap": len(turns) - len(eligible),
        "worst_turns": sorted(checks, key=lambda item: item["wer"], reverse=True)[:20],
    }


def evaluate_alignment(records: list[dict]) -> dict:
    flags = defaultdict(int)
    by_speaker = defaultdict(int)
    for record in records:
        by_speaker[record["speaker"]] += 1
        for flag in record.get("alignment_qc", {}).get("quality_flags", []):
            flags[flag] += 1
    reference_tokens = sum(record["alignment_qc"]["reference_token_count"] for record in records)
    aligned_words = sum(record["alignment_qc"]["aligned_word_count"] for record in records)
    return {
        "method": "CHAT_guided_WhisperX_word_alignment",
        "reviewed_turn_count": len(records),
        "reviewed_turn_count_by_CHAT_speaker": dict(by_speaker),
        "reference_token_count": reference_tokens,
        "aligned_word_count": aligned_words,
        "aligned_word_coverage": round(aligned_words / reference_tokens, 3)
        if reference_tokens else None,
        "quality_flag_counts": dict(flags),
        "timing_accuracy_requires_listening_review": True,
    }


def _dominant_cluster(start: float, end: float, segments: list[dict]) -> str | None:
    coverage: dict[str, float] = defaultdict(float)
    for segment in segments:
        overlap = max(0.0, min(end, segment["end"]) - max(start, segment["start"]))
        if overlap > 0:
            coverage[segment["speaker"]] += overlap
    return max(coverage, key=coverage.get) if coverage else None


def run_pilot(args: argparse.Namespace) -> dict:
    sources = args.source_id or list(DEFAULT_SOURCES)
    selected = []
    for source_id in sources:
        audio, chat = source_paths(args.root, source_id)
        duration = _audio_duration(audio)
        turns = all_timed_turns(chat, duration)
        sample = sample_turns(turns, args.turns_per_speaker)
        selected.append((source_id, audio, chat, duration, turns, sample))
    inventory = [{
        "source_id": source_id, "audio": str(audio), "chat": str(chat),
        "audio_duration_sec": round(duration, 3),
        "timed_turns": len(turns), "guided_alignment_turns": len(sample),
        "timed_turns_by_speaker": {
            tier: sum(turn.speaker == tier for turn in turns) for tier in sorted(TIERS)
        },
    } for source_id, audio, chat, duration, turns, sample in selected]
    if args.dry_run:
        return {"mode": "dry_run", "recordings": inventory}

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is missing. Put it in the Bengee project's ignored .env file.")
    from whisperx.diarize import DiarizationPipeline

    transcriber = GPUTranscriber(
        args.device, model_size=args.whisper_model,
        compute_type=args.compute_type, batch_size=args.batch_size,
        language="en", alignment_device=args.alignment_device,
    )
    diarizer = DiarizationPipeline(token=token, device=args.device)
    reports = []
    for source_id, audio, chat, duration, turns, sample in selected:
        work = args.out_dir / source_id.replace(":", "_")
        work.mkdir(parents=True, exist_ok=True)
        metadata = {
            "dataset": "NewmanRatner", "source_id": source_id,
            "source_chat": str(chat.resolve()),
            "speaker_reference": "CHAT_tier",
        }
        LOG.info("%s: aligning %d sampled CHAT turns", source_id, len(sample))
        guided_status = process_whisperx_chat_turns(
            sample, audio, work / "chat_guided_features.jsonl",
            transcriber=transcriber, batch_size=args.batch_size,
            pitch_floor=args.pitch_floor, pitch_ceiling=args.pitch_ceiling,
            whisper_model=args.whisper_model, compute_type=args.compute_type,
            alignment_device=args.alignment_device, metadata=metadata,
        )
        LOG.info("%s: transcribing full audio with Silero VAD", source_id)
        validate_independent_asr(
            turns, audio, work / "independent_asr_validation.json",
            transcriber=transcriber, whisper_model=args.whisper_model,
            compute_type=args.compute_type,
            alignment_device=args.alignment_device, metadata=metadata,
        )
        asr = json.loads((work / "independent_asr_validation.json").read_text(encoding="utf-8"))
        LOG.info("%s: diarizing full audio", source_id)
        diarized = diarizer(str(audio), min_speakers=2, max_speakers=4)
        diarization = [
            {"start": round(float(row.start), 3),
             "end": round(float(row.end), 3), "speaker": str(row.speaker)}
            for row in diarized.itertuples()
        ]
        import whisperx

        attributed = whisperx.assign_word_speakers(
            diarized, {"segments": asr["asr_segments"]}
        )
        attributed_segments = attributed["segments"]
        word_counts = defaultdict(int)
        for segment in attributed_segments:
            for word in segment.get("words", []):
                word_counts[word.get("speaker") or "unassigned"] += 1
        _write_json(work / "speaker_attributed_asr.json", {
            "schema_version": "newman_quality_speaker_asr_v1",
            **metadata, "source_audio": str(audio.resolve()),
            "speaker_label_provenance": "audio_diarization_cluster",
            "asr_transcript_provenance": "whisperx_silero_vad_independent",
            "word_count_by_cluster": dict(word_counts),
            "segments": attributed_segments,
        })
        _write_json(work / "diarization.json", {
            "schema_version": "newman_quality_diarization_v1",
            **metadata, "source_audio": str(audio.resolve()),
            "method": "whisperx_pyannote_community_1",
            "speaker_count_range": [2, 4],
            "segments": diarization,
        })
        records = [json.loads(line) for line in
                   (work / "chat_guided_features.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        report = {
            "schema_version": "newman_quality_pilot_v1",
            **metadata, "audio_duration_sec": round(duration, 3),
            "chat_turn_count": len(turns), "guided_alignment": evaluate_alignment(records),
            "independent_asr": evaluate_asr(turns, asr["asr_words"]),
            "speaker_diarization": evaluate_speakers(turns, diarization),
            "guided_status": guided_status,
            "artifacts": {
                "guided_features": str((work / "chat_guided_features.jsonl").resolve()),
                "independent_asr": str((work / "independent_asr_validation.json").resolve()),
                "diarization": str((work / "diarization.json").resolve()),
                "speaker_attributed_asr": str((work / "speaker_attributed_asr.json").resolve()),
            },
        }
        if not args.no_video:
            from tools.render_validation_video import render

            mapping = report["speaker_diarization"]["cluster_to_CHAT_mapping_for_scoring_only"]
            review_records = []
            for record in records:
                start, end = record["start_sec"], record["end_sec"]
                cluster = _dominant_cluster(start, end, diarization)
                asr_text = " ".join(
                    str(word["word"]) for word in asr["asr_words"]
                    if "start" in word and "end" in word
                    and start <= (word["start"] + word["end"]) / 2 < end
                )
                review_records.append({
                    **record, "diarization_speaker": cluster,
                    "diarization_mapped_speaker": mapping.get(cluster),
                    "independent_asr_text": asr_text,
                })
            videos = []
            for clip_index in range((int(duration - 0.001) // 600) + 1):
                clip_start = clip_index * 600.0
                if not any(record["start_sec"] < clip_start + 600.0
                           and record["end_sec"] > clip_start
                           for record in review_records):
                    continue
                video = work / f"review_video_{clip_index + 1:02d}.mp4"
                render(source_id, review_records, video, clip_start=clip_start)
                videos.append(str(video.resolve()))
            report["artifacts"]["review_videos"] = videos
        _write_json(work / "report.json", report)
        reports.append(report)
    summary = {"mode": "completed", "recordings": inventory,
               "reports": [report["source_id"] for report in reports]}
    _write_json(args.out_dir / "pilot_status.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/staging/newman_ratner"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/newman_quality_pilot"))
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--turns-per-speaker", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--alignment-device", default="cuda:0")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pitch-floor", type=float, default=75.0)
    parser.add_argument("--pitch-ceiling", type=float, default=600.0)
    parser.add_argument("--no-video", action="store_true",
                        help="Skip annotated audio/video review renders")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.turns_per_speaker < 1 or args.batch_size < 1:
        parser.error("turns per speaker and batch size must be positive")
    print(json.dumps(run_pilot(args), indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
