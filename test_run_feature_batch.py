from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_feature_batch import (
    FinalWriter, _ami_turns, _asr_turns, _audio_chunks,
    _load_speaker_segments, _manifest_rows, _record_signature,
    _selected_youtube_rows, _speaker_fields, _speaker_for_interval,
    _verify_audio_checksum,
)


class FeatureBatchTests(unittest.TestCase):
    def test_manifest_excludes_holds_and_missing_items(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.wav").write_bytes(b"a")
            manifest = root / "manifest.jsonl"
            rows = [
                {"dataset": "HVC", "source_id": "a", "relative_path": "a.wav",
                 "acquisition_status": "staged", "usage_tier": "acoustic_primary"},
                {"dataset": "HVC", "source_id": "b", "relative_path": "a.wav",
                 "acquisition_status": "staged", "usage_tier": "unpaired_hold"},
                {"dataset": "AMI", "source_id": "c", "relative_path": "missing.wav",
                 "acquisition_status": "missing", "usage_tier": "adult_supplement"},
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            selected = _manifest_rows(manifest, root, {"HVC", "AMI"})
            self.assertEqual([row["source_id"] for row in selected], ["a"])

    def test_ami_words_are_grouped_into_bounded_turns(self):
        with tempfile.TemporaryDirectory() as temp:
            xml = Path(temp) / "words.xml"
            xml.write_text(
                '<nite:root xmlns:nite="http://nite.sourceforge.net/">'
                '<w starttime="1.0" endtime="1.3">hello</w>'
                '<w starttime="1.3" endtime="1.3" punc="true">.</w>'
                '<w starttime="1.4" endtime="1.8">there</w>'
                '<w starttime="4.0" endtime="4.5">again</w>'
                '</nite:root>', encoding="utf-8",
            )
            turns = _ami_turns({"_annotation": xml, "speaker_id": "ES2002a.B"}, 25.0)
            self.assertEqual([turn.transcript for turn in turns], ["hello there", "again"])
            self.assertEqual(turns[0].annotations["manual_words"][0]["start"], 1.0)

    def test_selected_youtube_queue_contains_only_downloaded_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            (audio_dir / "abcdefghijk.m4a").write_bytes(b"audio")
            manifest = root / "selected.csv"
            manifest.write_text(
                "video_id,kind,yt_dlp_ok,title_discovered\n"
                "abcdefghijk,kids,True,Sample\n"
                "lmnopqrstuv,youtube,True,Missing\n", encoding="utf-8",
            )
            rows = _selected_youtube_rows(manifest, audio_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["candidate_kind"], "kids")
            self.assertIsNone(rows[0]["audience_label"])

    def test_speaker_ids_are_source_specific_and_unknowns_are_null(self):
        newman = {"dataset": "NewmanRatner", "age_group": "07",
                  "recording_id": "4269LP", "speaker_id": "MOT"}
        self.assertEqual(_speaker_fields(newman, "EXP")["speaker_id"],
                         "NewmanRatner:07:4269LP:EXP")
        self.assertEqual(_speaker_fields({"dataset": "AMI", "speaker_id": "ES2002a.B"})
                         ["speaker_attribution"], "headset_and_manual_word_annotation")
        self.assertIsNone(_speaker_fields({"dataset": "VoxPopuli", "speaker_id": "None"})
                          ["speaker_id"])
        self.assertIsNone(_speaker_fields({"dataset": "YouTube_selected_2000"})
                          ["speaker_id"])

    def test_asr_speaker_intervals_require_dominant_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "speakers.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in [
                {"source_id": "YouTube:abcdefghijk", "start_sec": 0, "end_sec": 9,
                 "speaker_id": "S1", "attribution": "manual_annotation"},
                {"source_id": "YouTube:abcdefghijk", "start_sec": 9, "end_sec": 20,
                 "speaker_id": "S2", "attribution": "external_diarization"},
            ]), encoding="utf-8")
            segments = _load_speaker_segments(path)["YouTube:abcdefghijk"]
            self.assertEqual(_speaker_for_interval(1, 8, segments),
                             ("S1", "manual_annotation"))
            self.assertEqual(_speaker_for_interval(8, 10, segments), (None, None))
            self.assertEqual(_speaker_for_interval(10, 12, segments),
                             ("S2", "external_diarization"))
            self.assertEqual(_speaker_fields(
                {"dataset": "YouTube_selected_2000", "source_id": "YouTube:abcdefghijk"},
                "S2", "external_diarization",
            )["speaker_id"], "YouTube:abcdefghijk:S2")

    def test_asr_decodes_only_short_chunks_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "long.mp3"
            audio.write_bytes(b"audio")
            args = argparse.Namespace(out_dir=root / "out", max_clip_sec=25.0,
                                      whisper_model="tiny", asr_device="cpu")
            row = {"dataset": "LibriVox", "source_id": "libri:test", "_audio": audio,
                   "_speaker_segments": [
                       {"start_sec": 0, "end_sec": 5, "speaker_id": "S1",
                        "attribution": "manual_annotation"},
                       {"start_sec": 50, "end_sec": 52, "speaker_id": "S2",
                        "attribution": "external_diarization"},
                   ]}
            calls = []

            def transcribe(clip):
                calls.append(clip)
                if len(calls) == 2:
                    raise ValueError("WhisperX produced no timed speech segments for MFA")
                return [{"text": "hello", "start": 1.0, "end": 2.0}], "en"

            def fake_transcriber(*args, **kwargs):
                return SimpleNamespace(transcribe_segments=transcribe)

            with patch("speech_pipeline.GPUTranscriber", side_effect=fake_transcriber), patch(
                "run_feature_batch._audio_duration", return_value=52.0
            ), patch("run_feature_batch._extract_clip"):
                first = _asr_turns(row, args)
                second = _asr_turns(row, args)
            self.assertEqual(len(calls), 3)
            self.assertEqual([turn.start_sec for turn in first], [1.0, 51.0])
            self.assertEqual([turn.speaker for turn in first], ["S1", "S2"])
            self.assertEqual(first, second)
            self.assertEqual(list(_audio_chunks(52.0, 25.0))[-1], (2, 50.0, 52.0))

    def test_final_jsonl_resume_and_source_change(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "features.jsonl"
            row = {"record_id": "HVC:one:seg_00000", "source_id": "HVC:one",
                   "source_signature": "abc"}
            self.assertEqual(FinalWriter(output).append([row]), 1)
            self.assertEqual(FinalWriter(output).append([row]), 0)
            with self.assertRaisesRegex(ValueError, "Source changed"):
                FinalWriter(output).append([{**row, "source_signature": "changed"}])
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 1)

    def test_signature_changes_with_transcript_and_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "segment.wav"
            audio.write_bytes(b"audio")
            row = {"dataset": "VoxPopuli", "source_id": "vox:1", "_audio": audio,
                   "_transcript": "hello"}
            args = argparse.Namespace(max_clip_sec=25.0, pitch_floor=75.0,
                                      pitch_ceiling=600.0, mfa_dictionary="english_us_mfa",
                                      mfa_acoustic_model="english_mfa", whisper_model="large-v3")
            first = _record_signature(row, args)
            row["_transcript"] = "goodbye"
            self.assertNotEqual(first, _record_signature(row, args))
            row["_transcript"] = "hello"
            args.pitch_ceiling = 800.0
            self.assertNotEqual(first, _record_signature(row, args))
            args.pitch_ceiling = 600.0
            row["_software_versions"] = {"parselmouth": "0.4.7"}
            self.assertNotEqual(first, _record_signature(row, args))

    def test_newman_signature_tracks_gpu_alignment_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "segment.wav"
            audio.write_bytes(b"audio")
            row = {"dataset": "NewmanRatner", "source_id": "NewmanRatner:07:4269LP:play",
                   "_audio": audio}
            args = argparse.Namespace(
                max_clip_sec=25.0, pitch_floor=75.0, pitch_ceiling=600.0,
                mfa_dictionary="english_us_mfa", mfa_acoustic_model="english_mfa",
                whisper_model="large-v3", newman_alignment_backend="whisperx_chat",
                newman_chat_context_sec=0.25, newman_validate_independent_asr=True,
                asr_device="cuda:0", alignment_device="cuda:0", compute_type="float16",
            )
            first = _record_signature(row, args)
            args.compute_type = "int8"
            self.assertNotEqual(first, _record_signature(row, args))
            args.compute_type = "float16"
            args.alignment_device = "cpu"
            self.assertNotEqual(first, _record_signature(row, args))

    def test_staged_audio_checksum_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "audio.wav"
            audio.write_bytes(b"audio")
            row = {"_audio": audio, "source_id": "sample",
                   "sha256": hashlib.sha256(b"audio").hexdigest()}
            _verify_audio_checksum(row)
            audio.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                _verify_audio_checksum(row)


if __name__ == "__main__":
    unittest.main()
