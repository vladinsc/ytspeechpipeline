from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mfa_alignment import Alignment
from newman_ratner_pipeline import (
    ChatTurn, _check_resume_config, chat_text_overlap, convert_chat,
    parse_chat, process_turns, process_whisperx_chat_turns,
    validate_independent_asr,
)


class NewmanRatnerPipelineTests(unittest.TestCase):
    def test_chat_conversion_keeps_timed_mother_speech_and_rejects_bad_links(self):
        with tempfile.TemporaryDirectory() as temp:
            chat = Path(temp) / "sample.cha"
            chat.write_text(
                "@UTF8\n"
                "*MOT:\t<which> [/] what is this ? \x151000_3000\x15\n"
                "%mor:\tignore this tier\n"
                "*EXP:\thello . \x153100_4000\x15\n"
                "*MOT:\tforty words cannot fit here . \x154000_4100\x15\n"
                "*MOT:\txxx a toy . \x155000_6000\x15\n"
                "*MOT:\tgood job\n\t little one . \x157000_9000\x15\n"
                "@End\n", encoding="utf-8"
            )
            turns, counts = convert_chat(chat)
            headers, all_turns = parse_chat(chat)

        self.assertEqual([t.transcript for t in turns], ["what is this", "good job little one"])
        self.assertEqual([t.start_sec for t in turns], [1.0, 7.0])
        self.assertEqual(counts["other_speaker"], 1)
        self.assertEqual(counts["implausible_word_rate"], 1)
        self.assertEqual(counts["uncertain_or_nonlexical"], 1)
        self.assertEqual(headers, ["@UTF8", "@End"])
        self.assertEqual(len(all_turns), 5)
        self.assertEqual(all_turns[0]["dependent_tiers"]["mor"], "ignore this tier")
        self.assertEqual(all_turns[1]["speaker"], "EXP")
        self.assertEqual(turns[0].dependent_tiers["mor"], "ignore this tier")

    def test_chat_overlap_ignores_different_time_links(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "a.cha", Path(temp) / "b.cha"
            first.write_text("*MOT:\thello there . \x151000_2000\x15\n", encoding="utf-8")
            second.write_text("*MOT:\thello there . \x1510000_11000\x15\n", encoding="utf-8")
            self.assertEqual(chat_text_overlap(first, second), 1.0)

    def test_alignment_batches_are_bounded_and_resume_without_asr(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            audio = base / "long.wav"
            audio.write_bytes(b"audio")
            output = base / "features.jsonl"
            turns = [
                ChatTurn(f"utt_{i:05d}", "MOT", float(10 + i), float(11 + i),
                         "hello", "hello .")
                for i in range(3)
            ]
            aligned = Alignment(
                words=[{"word": "hello", "start": 0.1, "end": 0.4}],
                phones=[{"phone": "HH", "start": 0.1, "end": 0.2, "word_index": 0}],
                output_path=base / "mfa.json",
            )
            batch_sizes = []

            def fake_align(corpus, output_root):
                names = [p.stem for p in corpus.glob("*.lab")]
                batch_sizes.append(len(names))
                return {name: aligned for name in names}

            def fake_analyzer(path, **kwargs):
                feature = SimpleNamespace(to_dict=lambda: {
                    "word": "hello", "start_time": 0.1, "end_time": 0.4,
                })
                syllable = SimpleNamespace(to_dict=lambda: {
                    "syllable": "hello", "start_time": 0.1, "end_time": 0.4,
                })
                return SimpleNamespace(
                    enrich=lambda words: [feature],
                    enrich_syllables=lambda words: [syllable],
                    sound=None, pitch=None,
                )

            with patch("newman_ratner_pipeline._audio_duration", return_value=30.0), patch(
                "newman_ratner_pipeline._run"
            ), patch("newman_ratner_pipeline.AcousticAnalyzer", side_effect=fake_analyzer), patch(
                "newman_ratner_pipeline.MFAAligner.ensure_available"
            ), patch("newman_ratner_pipeline.MFAAligner.align_corpus", side_effect=fake_align), patch(
                "newman_ratner_pipeline.summarize_sound", return_value={"word_count": 1}
            ):
                first = process_turns(turns, audio, output, batch_size=2,
                                      max_utterances=1)
                second = process_turns(turns, audio, output, batch_size=2)

            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(batch_sizes, [1, 2])
            self.assertEqual(first["completed_now"], 1)
            self.assertEqual(second["completed_now"], 2)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["transcript_source"], "CHAT")
            self.assertEqual(records[0]["language"], "en")
            self.assertEqual(records[0]["words"][0]["start_time"], 10.1)
            self.assertEqual(records[0]["phones"][0]["word_index"], 0)
            self.assertEqual(records[0]["acoustic_features"]["word_count"], 1)

    def test_resume_rejects_changed_transcript(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            audio, output = base / "audio.wav", base / "features.jsonl"
            audio.write_bytes(b"audio")
            first = [ChatTurn("utt_00001", "MOT", 0, 1, "hello", "hello")]
            changed = [ChatTurn("utt_00001", "MOT", 0, 1, "goodbye", "goodbye")]
            kwargs = dict(audio=audio, dictionary="english_us_mfa",
                          acoustic_model="english_mfa", pitch_floor=75.0,
                          pitch_ceiling=600.0, metadata=None)
            _check_resume_config(output, turns=first, **kwargs)
            output.write_text('{"utterance_id":"utt_00001"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Input or settings changed"):
                _check_resume_config(output, turns=changed, **kwargs)

    def test_chat_guided_whisperx_uses_context_but_summarizes_chat_window(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            audio, output = base / "audio.wav", base / "features.jsonl"
            audio.write_bytes(b"audio")
            turn = ChatTurn("utt_00001", "MOT", 2.0, 3.0, "hello there", "hello there .")
            calls = []

            class FakeTranscriber:
                def align_known_transcript(self, clip, transcript, **kwargs):
                    calls.append((clip, transcript, kwargs))
                    return [
                        {"word": "hello", "start": 0.3, "end": 0.5},
                        {"word": "there", "start": 0.6, "end": 0.85},
                    ]

            def fake_analyzer(path, **kwargs):
                def word(word):
                    return SimpleNamespace(to_dict=lambda: {
                        "word": word["word"], "start_time": word["start"],
                        "end_time": word["end"],
                    })
                return SimpleNamespace(
                    enrich=lambda words: [word(item) for item in words],
                    enrich_syllables=lambda words: [], sound=None, pitch=None,
                )

            with patch("newman_ratner_pipeline._audio_duration", return_value=5.0), patch(
                "newman_ratner_pipeline._run"
            ), patch("newman_ratner_pipeline.AcousticAnalyzer", side_effect=fake_analyzer), patch(
                "newman_ratner_pipeline.summarize_sound", return_value={"duration_sec": 1.0}
            ) as summary:
                result = process_whisperx_chat_turns(
                    [turn], audio, output, transcriber=FakeTranscriber(),
                    chat_context_sec=0.25,
                )

            record = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["completed_now"], 1)
            self.assertEqual(calls[0][1], "hello there")
            self.assertEqual(calls[0][2]["start_sec"], 0.0)
            self.assertEqual(calls[0][2]["end_sec"], 1.5)
            self.assertEqual(record["alignment_method"], "whisperx_chat_guided")
            self.assertEqual(record["phones"], [])
            self.assertEqual(record["words"][0]["start_time"], 2.05)
            self.assertEqual(summary.call_args.kwargs["window_start"], 0.25)
            self.assertEqual(summary.call_args.kwargs["window_end"], 1.25)

    def test_independent_asr_validation_keeps_chat_and_asr_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            audio, output = base / "audio.wav", base / "validation.json"
            audio.write_bytes(b"audio")
            turns = [ChatTurn("utt_00001", "MOT", 1.0, 2.0, "hello there", "hello there .")]

            class FakeTranscriber:
                def transcribe_with_alignment(self, path):
                    return {
                        "language": "en",
                        "words": [{"word": "hello", "start": 1.1, "end": 1.4}],
                        "segments": [{"text": "hello there", "start": 1.0, "end": 2.0, "words": []}],
                    }

            with patch("newman_ratner_pipeline._audio_duration", return_value=3.0), patch(
                "newman_ratner_pipeline._run"
            ):
                result = validate_independent_asr(turns, audio, output, transcriber=FakeTranscriber())

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result["target_speaker_wer"], 0.0)
            self.assertEqual(payload["validation_purpose"], "diagnostic_only_does_not_modify_CHAT_annotation")
            self.assertEqual(payload["chat_turn_checks"][0]["chat_transcript"], "hello there")


if __name__ == "__main__":
    unittest.main()
