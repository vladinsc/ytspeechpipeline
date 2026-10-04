import argparse
import json
import math
import tempfile
import unittest
import wave
from pathlib import Path

from run_transcription_batch import assign_shards, inventory, transcribe_row
from tools.build_transcription_manifest import build_rows


class TranscriptionBatchTests(unittest.TestCase):
    def test_selected_youtube_manifest_deduplicates_and_excludes_known_non_english(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_dir = root / "audio"
            audio_dir.mkdir()
            for video_id, language in (("abcdefghijk", "en"), ("lmnopqrstuv", "hi")):
                (audio_dir / f"{video_id}.m4a").write_bytes(b"sample audio")
                (audio_dir / f"{video_id}.info.json").write_text(
                    json.dumps({"duration": 60, "language": language, "title": video_id}),
                    encoding="utf-8")
            base = root / "base.jsonl"
            base.write_text("", encoding="utf-8")
            selection = root / "selection.csv"
            selection.write_text("video_id,kind\nabcdefghijk,kids\nabcdefghijk,youtube\n"
                                 "lmnopqrstuv,kids\n", encoding="utf-8")
            rows, counts = build_rows(base, selection, audio_dir, root)
            self.assertEqual(counts["included_audio"], 1)
            self.assertEqual(counts["known_non_english"], 1)
            self.assertEqual(rows[0]["candidate_kinds"], ["kids", "youtube"])
            self.assertEqual(rows[0]["audience_label"], None)

    def test_inventory_excludes_text_and_holds_and_shards_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "one.wav").write_bytes(b"audio")
            (root / "two.wav").write_bytes(b"audio")
            items = [
                {"dataset": "LibriVox", "source_id": "one", "acquisition_status": "staged",
                 "transcript_status": "whisperx_required", "language": "en",
                 "usage_tier": "narration_supplement", "rights_review": "accepted",
                 "relative_path": "one.wav", "duration_seconds": 20},
                {"dataset": "LibriVox", "source_id": "two", "acquisition_status": "staged",
                 "transcript_status": "whisperx_required", "language": "en",
                 "usage_tier": "narration_supplement", "rights_review": "accepted",
                 "relative_path": "two.wav", "duration_seconds": 10},
                {"dataset": "AMI", "source_id": "text", "acquisition_status": "staged",
                 "transcript_status": "manual_word_timestamps", "language": "en",
                 "relative_path": "one.wav"},
                {"dataset": "YouTube_SciShow_pilot", "source_id": "hold",
                 "acquisition_status": "staged", "transcript_status": "whisperx_required",
                 "language": "en", "usage_tier": "exploratory_hold",
                 "relative_path": "one.wav"},
            ]
            manifest = root / "manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(item) for item in items), encoding="utf-8")
            rows, summary = inventory(manifest, root, {"LibriVox"}, False)
            self.assertEqual(len(rows), 2)
            self.assertEqual(summary["AMI"]["has_transcript"], 1)
            self.assertEqual(summary["YouTube_SciShow_pilot"]["hold"], 1)
            shards = assign_shards(rows, 3)
            self.assertEqual({r["source_id"] for shard in shards for r in shard}, {"one", "two"})

    def test_transcript_output_is_segment_timed_and_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            with wave.open(str(audio), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16000)
                output.writeframes(b"\x00\x00" * 16000)
            row = {"source_id": "LibriVox:sample", "dataset": "LibriVox",
                   "audio": audio, "relative_path": "sample.wav",
                   "expected_sha256": "", "usage_tier": "narration_supplement",
                   "audience_label": "kids_directed"}
            args = argparse.Namespace(out_dir=root / "results", model="stub",
                                      compute_type="float16", batch_size=1,
                                      max_clip_sec=25)

            class Stub:
                def transcribe_segments(self, clip):
                    return ([{"text": "  hello   world  ", "start": 0.1,
                              "end": 0.8}], "en")

            self.assertEqual(transcribe_row(row, args, lambda: Stub()), "completed")
            self.assertEqual(transcribe_row(row, args, lambda: None), "skipped")
            output = next((args.out_dir / "transcripts").glob("*.json"))
            data = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(data["timing_level"], "asr_segment_only")
            self.assertEqual(data["segments"][0]["text"], "hello world")
            self.assertTrue(math.isclose(data["segments"][0]["start_sec"], 0.1))


if __name__ == "__main__":
    unittest.main()
