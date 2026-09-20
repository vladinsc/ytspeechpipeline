from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from batch_pipeline import parse_args
from speech_pipeline import GPUTranscriber


class _FakeWhisperModel:
    def __init__(self, languages: list[str]):
        self.languages = iter(languages)
        self.batch_sizes: list[int] = []

    def transcribe(self, _audio, batch_size: int, language: str | None):
        self.batch_sizes.append(batch_size)
        return {"language": next(self.languages), "segments": []}


class TranscriberCacheTests(unittest.TestCase):
    def fake_whisperx(self, languages: list[str]):
        whisper_model = _FakeWhisperModel(languages)
        align_loads: list[tuple[str, str]] = []

        def load_align_model(language_code: str, device: str):
            align_loads.append((language_code, device))
            return object(), {"language": language_code}

        module = SimpleNamespace(
            load_model=lambda *_args, **_kwargs: whisper_model,
            load_audio=lambda _path: object(),
            load_align_model=load_align_model,
            align=lambda *_args, **_kwargs: {"segments": []},
        )
        return module, whisper_model, align_loads

    def test_batch_parser_accepts_high_memory_gpu_sizes(self) -> None:
        self.assertEqual(parse_args(["videos.txt", "--batch-size", "8"]).batch_size, 8)
        self.assertEqual(parse_args(["videos.txt", "--batch-size", "16"]).batch_size, 16)

        with self.assertRaises(SystemExit):
            parse_args(["videos.txt", "--batch-size", "0"])

    def test_alignment_model_is_reused_for_the_same_language(self) -> None:
        fake, whisper_model, align_loads = self.fake_whisperx(["en", "en"])
        with patch.dict(sys.modules, {"whisperx": fake}):
            transcriber = GPUTranscriber("cpu", batch_size=16, language="en")
            transcriber.transcribe("first.wav")
            transcriber.transcribe("second.wav")

        self.assertEqual(align_loads, [("en", "cpu")])
        self.assertEqual(whisper_model.batch_sizes, [16, 16])

    def test_alignment_cache_is_replaced_when_language_changes(self) -> None:
        fake, _whisper_model, align_loads = self.fake_whisperx(["en", "fr", "fr"])
        with patch.dict(sys.modules, {"whisperx": fake}):
            transcriber = GPUTranscriber("cpu")
            transcriber.transcribe("english.wav")
            transcriber.transcribe("french-1.wav")
            transcriber.transcribe("french-2.wav")

        self.assertEqual(align_loads, [("en", "cpu"), ("fr", "cpu")])


if __name__ == "__main__":
    unittest.main()
