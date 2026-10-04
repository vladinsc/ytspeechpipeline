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

    def test_full_review_shows_chat_asr_and_both_word_tracks(self):
        record = {
            "dataset": "NewmanRatner", "start_sec": 1.0, "end_sec": 2.0,
            "speaker": "MOT", "transcript": "hello", "words": [
                {"word": "hello", "start_time": 1.2, "end_time": 1.7},
            ],
        }
        ass, counts = build_ass(
            [record], "NewmanRatner:18:sample:interview", 0.0, 5.0,
            chat_turns=[
                {"speaker": "MOT", "start": 1.0, "end": 2.0, "text": "hello"},
                {"speaker": "CHI", "start": 1.5, "end": 2.5, "text": "yes"},
            ],
            asr_segments=[{"start": 1.0, "end": 2.5, "text": "hello yes"}],
            asr_words=[{"start": 1.1, "end": 1.6, "word": "hello"},
                       {"start": 1.8, "end": 2.3, "word": "yes"}],
            diarization_segments=[{"start": 1.0, "end": 2.5,
                                   "speaker": "SPEAKER_00"}],
        )
        self.assertIn("CHAT: MOT: hello | CHI: yes", ass)
        self.assertIn("WhisperX transcript: hello yes", ass)
        self.assertIn("WhisperX aligned word: [hello] yes", ass)
        self.assertIn("CHAT-guided aligned word: [hello]", ass)
        self.assertIn("Audio speaker cluster: SPEAKER_00", ass)
        self.assertIn("CHAT: no timed transcript for this span", ass)
        self.assertEqual(counts["independent_asr_words_with_timing"], 2)
        self.assertEqual(counts["timed_chat_turns_available"], 2)


if __name__ == "__main__":
    unittest.main()
