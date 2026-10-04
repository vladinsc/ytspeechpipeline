from __future__ import annotations

import unittest

from newman_ratner_pipeline import ChatTurn
from tools.run_newman_quality_pilot import (
    evaluate_alignment, evaluate_asr, evaluate_speakers, sample_turns,
)


class NewmanQualityPilotTests(unittest.TestCase):
    def test_speaker_clusters_are_mapped_only_for_scoring(self):
        turns = [
            ChatTurn("u1", "MOT", 0, 2, "hello there", "hello there"),
            ChatTurn("u2", "EXP", 2, 4, "yes", "yes"),
            ChatTurn("u3", "CHI", 4, 5, "no", "no"),
        ]
        segments = [
            {"start": 0, "end": 2, "speaker": "SPEAKER_02"},
            {"start": 2, "end": 4, "speaker": "SPEAKER_00"},
            {"start": 4, "end": 5, "speaker": "SPEAKER_01"},
        ]
        report = evaluate_speakers(turns, segments)
        self.assertEqual(report["speaker_coverage"], 1.0)
        self.assertEqual(report["mapped_accuracy_on_covered_speech"], 1.0)
        self.assertEqual(report["cluster_to_CHAT_mapping_for_scoring_only"]["SPEAKER_02"], "MOT")

    def test_asr_comparison_uses_word_midpoints_and_excludes_cross_speaker_overlap(self):
        turns = [
            ChatTurn("u1", "MOT", 0, 2, "hello there", "hello there"),
            ChatTurn("u2", "EXP", 1.9, 3, "yes", "yes"),
            ChatTurn("u3", "MOT", 4, 5, "goodbye", "goodbye"),
        ]
        words = [{"word": "hello", "start": 0.1, "end": 0.5},
                 {"word": "there", "start": 0.6, "end": 1.1},
                 {"word": "goodbye", "start": 4.1, "end": 4.6}]
        report = evaluate_asr(turns, words)
        self.assertEqual(report["turns_excluded_for_cross_speaker_overlap"], 2)
        self.assertEqual(report["turns_scored"], 1)
        self.assertEqual(report["diagnostic_wer"], 0.0)

    def test_sample_spreads_across_each_speaker_and_alignment_coverage_is_visible(self):
        turns = [ChatTurn(f"m{i}", "MOT", i, i + 0.5, "hello", "hello")
                 for i in range(10)]
        turns += [ChatTurn(f"e{i}", "EXP", i, i + 0.5, "yes", "yes")
                  for i in range(3)]
        sample = sample_turns(turns, 3)
        self.assertEqual({turn.utterance_id for turn in sample if turn.speaker == "MOT"},
                         {"m0", "m4", "m9"})
        records = [{"speaker": "MOT", "alignment_qc": {"reference_token_count": 10,
                                      "aligned_word_count": 8,
                                      "quality_flags": ["word_outside_CHAT_time_link"]}}]
        report = evaluate_alignment(records)
        self.assertEqual(report["aligned_word_coverage"], 0.8)
        self.assertEqual(report["quality_flag_counts"]["word_outside_CHAT_time_link"], 1)


if __name__ == "__main__":
    unittest.main()
