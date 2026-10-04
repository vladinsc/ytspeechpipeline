from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from batch_pipeline import parse_args as parse_batch_args
from mfa_alignment import Alignment, MFAAligner, parse_mfa_json
from speech_pipeline import PipelineConfig, ProsodyPipeline, parse_args


EXAMPLE_TIERS = {
    "tiers": [
        {"name": "speaker - words", "entries": [
            [0.0, 0.2, ""], [0.2, 0.6, "Hello"], [0.7, 1.0, "world"],
        ]},
        {"name": "speaker - phones", "entries": [
            [0.2, 0.3, "HH"], [0.3, 0.6, "AH0"],
            [0.7, 0.8, "W"], [0.8, 1.0, "ER1"],
        ]},
    ]
}


class MFAAlignmentTests(unittest.TestCase):
    def test_mfa_is_default_alignment_backend(self):
        self.assertEqual(parse_args(["segment.wav"]).alignment_backend, "mfa")
        self.assertEqual(parse_batch_args(["videos.txt"]).alignment_backend, "mfa")

    def test_parses_word_and_phone_tiers(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "aligned.json"
            path.write_text(json.dumps(EXAMPLE_TIERS), encoding="utf-8")
            aligned = parse_mfa_json(path)

        self.assertEqual([w["word"] for w in aligned.words], ["Hello", "world"])
        self.assertEqual([p["phone"] for p in aligned.phones], ["HH", "AH0", "W", "ER1"])
        self.assertEqual([p["word_index"] for p in aligned.phones], [0, 0, 1, 1])

    def test_parses_current_mfa_named_tier_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "aligned.json"
            path.write_text(json.dumps({"tiers": {
                "words": {"type": "IntervalTier", "entries": [[0.0, 0.2, "sure"]]},
                "phones": {"type": "IntervalTier", "entries": [[0.0, 0.2, "ʃ"]]},
            }}), encoding="utf-8")
            aligned = parse_mfa_json(path)
        self.assertEqual(aligned.words[0]["word"], "sure")
        self.assertEqual(aligned.phones[0]["phone"], "ʃ")
        self.assertEqual(aligned.phones[0]["word_index"], 0)

    def test_rejects_missing_phone_tier(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "aligned.json"
            path.write_text(json.dumps({"tiers": EXAMPLE_TIERS["tiers"][:1]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing word or phone"):
                parse_mfa_json(path)

    def test_aligner_uses_fresh_output_and_expected_cli(self):
        with tempfile.TemporaryDirectory() as temp:
            workdir = Path(temp)
            audio = workdir / "segment.wav"
            transcript = workdir / "segment.txt"
            output = workdir / "mfa_alignment.json"
            audio.write_bytes(b"wav")
            transcript.write_text("Hello world", encoding="utf-8")
            output.write_text("stale", encoding="utf-8")

            def fake_run(command, **kwargs):
                self.assertFalse(output.exists())
                self.assertEqual(command[:4], ["mfa", "align_one", "--output_format", "json"])
                self.assertEqual(command[4], "-t")
                self.assertEqual(command[6:], [str(audio), str(transcript), "english_mfa", "english_mfa", str(output)])
                output.write_text(json.dumps(EXAMPLE_TIERS), encoding="utf-8")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("mfa_alignment.shutil.which", return_value="mfa"), patch(
                "mfa_alignment.subprocess.run", side_effect=fake_run
            ):
                result = MFAAligner("english_mfa", "english_mfa").align(audio, transcript, workdir)
            self.assertEqual(len(result.words), 2)
            self.assertEqual(len(result.phones), 4)

    def test_aligner_can_launch_mfa_in_separate_conda_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            workdir = Path(temp)
            audio = workdir / "segment.wav"
            transcript = workdir / "segment.txt"
            audio.write_bytes(b"wav")
            transcript.write_text("Hello world", encoding="utf-8")

            def fake_run(command, **kwargs):
                self.assertEqual(command[:8], [
                    "conda", "run", "-n", "aligner", "mfa", "align_one",
                    "--output_format", "json",
                ])
                self.assertEqual(command[8], "-t")
                self.assertEqual(command[10], str(audio))
                (workdir / "mfa_alignment.json").write_text(
                    json.dumps(EXAMPLE_TIERS), encoding="utf-8"
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("mfa_alignment.shutil.which", return_value="conda"), patch(
                "mfa_alignment.subprocess.run", side_effect=fake_run
            ):
                result = MFAAligner("english_us_mfa", "english_mfa", conda_env="aligner").align(
                    audio, transcript, workdir
                )
            self.assertEqual(len(result.words), 2)

    def test_corpus_alignment_uses_one_mfa_invocation(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            corpus = base / "corpus"
            corpus.mkdir()
            for stem in ("clip_00000", "clip_00001"):
                (corpus / f"{stem}.wav").write_bytes(b"wav")
                (corpus / f"{stem}.lab").write_text("Hello world", encoding="utf-8")

            def fake_run(command, **kwargs):
                self.assertEqual(command[:4], ["mfa", "align", "--output_format", "json"])
                self.assertEqual(command[4], "-t")
                self.assertEqual(command[6], str(corpus))
                output = Path(command[-1])
                for stem in ("clip_00000", "clip_00001"):
                    (output / f"{stem}.json").write_text(
                        json.dumps(EXAMPLE_TIERS), encoding="utf-8"
                    )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with patch("mfa_alignment.shutil.which", return_value="mfa"), patch(
                "mfa_alignment.subprocess.run", side_effect=fake_run
            ) as run:
                aligned = MFAAligner("english_us_mfa", "english_mfa").align_corpus(
                    corpus, base / "out"
                )
            run.assert_called_once()
            self.assertEqual(set(aligned), {"clip_00000", "clip_00001"})
            self.assertEqual(len(aligned["clip_00001"].phones), 4)

    def test_local_transcript_uses_mfa_without_gpu_or_asr(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            audio = base / "recording.wav"
            transcript = base / "recording.txt"
            audio.write_bytes(b"source")
            transcript.write_text("Hello world", encoding="utf-8")
            cfg = PipelineConfig(
                url=str(audio), out_path=base / "features.json", workdir=base / "work",
                transcript_path=transcript, mfa_dictionary="english_mfa",
                mfa_acoustic_model="english_mfa", language="en",
            )
            aligned = Alignment(
                words=[{"word": "Hello", "start": 0.2, "end": 0.6}],
                phones=[{"phone": "HH", "start": 0.2, "end": 0.3, "word_index": 0}],
                output_path=base / "aligned.json",
            )
            analyzer = SimpleNamespace(enrich=lambda words: [SimpleNamespace(
                to_dict=lambda: {"word": words[0]["word"], "pitch_mean_hz": 200.0}
            )])
            with patch("speech_pipeline._run") as run_command, patch(
                "speech_pipeline.AcousticAnalyzer", return_value=analyzer
            ) as analyzer_class, patch(
                "mfa_alignment.MFAAligner.align", return_value=aligned
            ) as align, patch("mfa_alignment.MFAAligner.ensure_available"), patch(
                "speech_pipeline.GPUTranscriber"
            ) as asr:
                result = ProsodyPipeline(cfg).run()

            self.assertEqual(run_command.call_count, 2)
            self.assertNotIn("-ar", run_command.call_args_list[0].args[0])
            self.assertEqual(analyzer_class.call_args.args[0].name, "local_analysis_mono.wav")
            self.assertEqual(align.call_args.args[1], transcript)
            asr.assert_not_called()
            self.assertEqual(result["alignment_method"], "mfa")
            self.assertEqual(result["transcript_source"], "provided")
            self.assertEqual(result["phones"][0]["word_index"], 0)
            self.assertEqual(json.loads(cfg.out_path.read_text(encoding="utf-8")), result)

    def test_asr_segments_are_aligned_with_mfa_and_shifted_to_source_times(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            audio = base / "recording.wav"
            audio.write_bytes(b"source")
            cfg = PipelineConfig(
                url=str(audio), out_path=base / "features.json", workdir=base / "work",
                language="en",
            )
            segments = [
                {"text": "hello", "start": 1.0, "end": 1.5},
                {"text": "world", "start": 3.0, "end": 3.5},
            ]
            transcriber = SimpleNamespace(transcribe_segments=lambda _audio: (segments, "en"))
            aligned = Alignment(
                words=[{"word": "word", "start": 0.1, "end": 0.4}],
                phones=[{"phone": "W", "start": 0.1, "end": 0.2, "word_index": 0}],
                output_path=base / "aligned.json",
            )
            analyzer = SimpleNamespace(enrich=lambda words: [SimpleNamespace(
                to_dict=lambda word=word: {"word": word["word"]}
            ) for word in words])
            with patch("speech_pipeline._run") as run_command, patch(
                "speech_pipeline.AcousticAnalyzer", return_value=analyzer
            ), patch("mfa_alignment.MFAAligner.ensure_available"), patch(
                "mfa_alignment.MFAAligner.align_corpus",
                return_value={"clip_00000": aligned, "clip_00001": aligned},
            ) as align:
                result = ProsodyPipeline(cfg, device="cpu").run(transcriber=transcriber)

            self.assertEqual(run_command.call_count, 4)
            align.assert_called_once()
            self.assertEqual([phone["start"] for phone in result["phones"]], [1.1, 3.1])
            self.assertEqual([phone["word_index"] for phone in result["phones"]], [0, 1])
            self.assertEqual(result["transcript_source"], "whisperx_asr")


if __name__ == "__main__":
    unittest.main()
