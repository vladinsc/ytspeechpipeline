from __future__ import annotations

import unittest

from tools.render_validation_video import build_ass


class ValidationVideoTests(unittest.TestCase):
    def test_timed_overlays_show_words_phones_speaker_and_features(self):
        row = {
            "dataset": "AMI", "source_id": "AMI:sample", "start_sec": 11.0,
            "end_sec": 13.0, "speaker_id": "meeting.B",
            "speaker_attribution": "headset_and_manual_word_annotation",
            "transcript": "hello", "quality_flags": ["test_flag"],
            "acoustic_features": {"pitch_median_hz": 210, "pitch_p05_hz": 190,
                                  "pitch_p95_hz": 230, "intensity_mean_db": 72,
                                  "words_per_sec": 0.5, "voiced_frame_fraction": 0.8},
            "words": [{"word": "hello", "start_time": 11.2, "end_time": 11.6,
                       "pitch_mean_hz": 211, "pause_after_sec": None}],
            "phones": [{"phone": "HH", "start": 11.2, "end": 11.3,
                        "word_index": 0}],
        }
        ass, counts = build_ass([row], "AMI:sample", 10.0, 10.0)
        self.assertIn("Speaker: meeting.B", ass)
        self.assertIn("Transcript: hello", ass)
        self.assertIn("Aligned word: [hello]", ass)
        self.assertIn("Aligned phone: HH", ass)
        self.assertIn("F0 median 210 Hz", ass)
        self.assertIn("QC: test_flag", ass)
        self.assertIn("0:00:01.20,0:00:01.60,Word", ass)
        self.assertEqual(counts["word_events"], 1)
        self.assertEqual(counts["phone_events"], 1)

    def test_hvc_reports_unavailable_transcript_and_alignment(self):
        row = {"dataset": "HVC", "source_id": "HVC:sample",
               "start_sec": 0.0, "end_sec": 3.0,
               "speaker_id": "sample", "speaker_attribution": "paired_recording_manifest",
               "transcript": None, "acoustic_features": {"pitch_median_hz": 205},
               "quality_flags": ["no_lexical_transcript"],
               "words": [], "phones": []}
        ass, counts = build_ass([row], "HVC:sample", 0.0, 3.0)
        self.assertIn("No lexical transcript supplied", ass)
        self.assertIn("Word alignment unavailable", ass)
        self.assertIn("Phone alignment unavailable", ass)
        self.assertEqual(counts["word_events"], 0)

    def test_later_clip_uses_source_timeline_with_local_subtitle_times(self):
        row = {"dataset": "NewmanRatner", "source_id": "NewmanRatner:24:sample:interview",
               "start_sec": 610.0, "end_sec": 612.0, "speaker": "CHI",
               "transcript": "hello", "independent_asr_text": "hello",
               "diarization_speaker": "SPEAKER_02", "words": [], "phones": []}
        ass, counts = build_ass([row], row["source_id"], 600.0, 30.0)
        self.assertIn("0:00:10.00,0:00:12.00,Speaker", ass)
        self.assertIn("audio cluster: SPEAKER_02", ass)
        self.assertIn("Independent ASR: hello", ass)
        self.assertEqual(counts["shown_utterances_or_clips"], 1)


if __name__ == "__main__":
    unittest.main()
