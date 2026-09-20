from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speech_pipeline import AudioDownloader


class DownloaderCookieTests(unittest.TestCase):
    def run_download(self, workdir: Path) -> list[list[str]]:
        downloader = AudioDownloader.__new__(AudioDownloader)
        downloader.workdir = workdir
        commands: list[list[str]] = []

        def fake_run(command: list[str]) -> None:
            commands.append(command)
            if command[0] == "yt-dlp":
                (workdir / "download.m4a").write_bytes(b"audio")

        with patch("speech_pipeline._run", side_effect=fake_run):
            downloader.download("https://www.youtube.com/watch?v=test")
        return commands

    def test_cookie_file_is_passed_to_yt_dlp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            cookies = workdir / "cookies.txt"
            cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
            with patch.dict(
                os.environ, {"YT_DLP_COOKIES_FILE": str(cookies)}, clear=False
            ):
                commands = self.run_download(workdir)

        self.assertIn("--cookies", commands[0])
        self.assertEqual(commands[0][commands[0].index("--cookies") + 1], str(cookies))

    def test_cookie_option_is_omitted_when_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {}, clear=True):
                commands = self.run_download(Path(directory))

        self.assertNotIn("--cookies", commands[0])

    def test_missing_configured_cookie_file_fails_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory)
            missing = workdir / "missing-cookies.txt"
            downloader = AudioDownloader.__new__(AudioDownloader)
            downloader.workdir = workdir
            with patch.dict(
                os.environ, {"YT_DLP_COOKIES_FILE": str(missing)}, clear=False
            ), patch("speech_pipeline._run") as run:
                with self.assertRaisesRegex(FileNotFoundError, "YT_DLP_COOKIES_FILE"):
                    downloader.download("https://www.youtube.com/watch?v=test")
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
