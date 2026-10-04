"""Run Montreal Forced Aligner on one transcribed speech segment.

MFA is installed separately from the GPU pipeline. This module only invokes its
CLI and converts the resulting word/phone tiers to the pipeline's JSON schema.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class Alignment:
    words: list[dict]
    phones: list[dict]
    output_path: Path


@lru_cache(maxsize=16)
def _verify_mfa_command(prefix: tuple[str, ...]) -> None:
    result = subprocess.run(
        [*prefix, "--help"], capture_output=True, text=True,
        check=False, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MFA is unavailable (exit {result.returncode}): "
            f"{result.stderr[-1000:] or result.stdout[-1000:]}"
        )


def _intervals(tier: dict, label_key: str) -> list[dict]:
    intervals = []
    for entry in tier.get("entries", []):
        if len(entry) != 3:
            raise ValueError(f"Invalid MFA {label_key} interval: {entry!r}")
        start, end, label = entry
        start, end = float(start), float(end)
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f"Invalid MFA {label_key} times: {entry!r}")
        label = str(label).strip()
        if label:
            intervals.append({label_key: label, "start": start, "end": end})
    return intervals


def parse_mfa_json(path: Path) -> Alignment:
    """Read MFA's Praat-compatible JSON tiers, requiring words and phones."""
    data = json.loads(path.read_text(encoding="utf-8"))
    tiers = data.get("tiers")
    if isinstance(tiers, dict):
        # MFA 3.4 writes Praat-compatible JSON as a name -> tier mapping.
        tiers = [{"name": name, **tier} for name, tier in tiers.items()]
    if not isinstance(tiers, list):
        raise ValueError(f"MFA output has no tier collection: {path}")
    by_name = {}
    for tier in tiers:
        name = str(tier.get("name", "")).lower().strip()
        # MFA may prefix a tier with the speaker ID.
        for kind in ("words", "phones"):
            if name == kind or name.endswith(f" - {kind}"):
                by_name.setdefault(kind, []).append(tier)
    if not by_name.get("words") or not by_name.get("phones"):
        raise ValueError(f"MFA output is missing word or phone tiers: {path}")
    words = sorted(
        (item for tier in by_name["words"] for item in _intervals(tier, "word")),
        key=lambda item: item["start"],
    )
    phones = sorted(
        (item for tier in by_name["phones"] for item in _intervals(tier, "phone")),
        key=lambda item: item["start"],
    )
    if not words or not phones:
        raise ValueError(f"MFA produced no aligned words or phones: {path}")
    for phone in phones:
        midpoint = (phone["start"] + phone["end"]) / 2
        phone["word_index"] = next(
            (i for i, word in enumerate(words) if word["start"] <= midpoint <= word["end"]),
            None,
        )
    return Alignment(words=words, phones=phones, output_path=path)


class MFAAligner:
    def __init__(
        self, dictionary: str, acoustic_model: str, executable: str = "mfa",
        conda_env: str | None = None, config_path: Path | None = None,
    ):
        if not dictionary or not acoustic_model:
            raise ValueError("MFA needs a pronunciation dictionary and acoustic model")
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.executable = executable
        self.conda_env = conda_env
        if config_path is not None and not config_path.is_file():
            raise FileNotFoundError(f"MFA configuration file not found: {config_path}")
        self.config_path = config_path

    def _config_arguments(self) -> list[str]:
        return ["--config_path", str(self.config_path)] if self.config_path else []

    def _command_prefix(self) -> list[str]:
        launcher = "conda" if self.conda_env else self.executable
        binary = shutil.which(launcher)
        if binary is None:
            raise FileNotFoundError(
                f"{launcher!r} executable not found. Install MFA and make its "
                "environment available on PATH or set --mfa-conda-env."
            )
        return ([binary, "run", "-n", self.conda_env, "mfa"]
                if self.conda_env else [binary])

    def ensure_available(self) -> None:
        """Fail before audio preparation if the MFA launcher is unavailable."""
        _verify_mfa_command(tuple(self._command_prefix()))

    def align(self, audio: Path, transcript: Path, workdir: Path) -> Alignment:
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if not transcript.is_file():
            raise FileNotFoundError(transcript)
        if not transcript.read_text(encoding="utf-8-sig").strip():
            raise ValueError(f"Transcript is empty: {transcript}")
        prefix = self._command_prefix()
        workdir.mkdir(parents=True, exist_ok=True)
        output = workdir / "mfa_alignment.json"
        temp_dir = workdir / "mfa_temporary"
        temp_dir.mkdir(parents=True, exist_ok=True)
        # A previous run must never be mistaken for this run's result.
        output.unlink(missing_ok=True)
        command = [
            *prefix, "align_one", *self._config_arguments(),
            "--output_format", "json", "-t", str(temp_dir),
            str(audio), str(transcript), self.dictionary,
            self.acoustic_model, str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"MFA alignment failed (exit {result.returncode}): "
                f"{result.stderr[-1500:] or result.stdout[-1500:]}"
            )
        if not output.is_file():
            raise RuntimeError(f"MFA did not create alignment output: {output}")
        return parse_mfa_json(output)

    def align_corpus(self, corpus_dir: Path, output_root: Path) -> dict[str, Alignment]:
        """Align all matching audio/transcript clips in one MFA invocation."""
        transcripts = sorted(corpus_dir.glob("*.lab"))
        if not transcripts:
            raise ValueError(f"No transcript clips in {corpus_dir}")
        for transcript in transcripts:
            if not (corpus_dir / f"{transcript.stem}.wav").is_file():
                raise FileNotFoundError(f"Missing audio for {transcript}")
            if not transcript.read_text(encoding="utf-8-sig").strip():
                raise ValueError(f"Transcript is empty: {transcript}")
        prefix = self._command_prefix()
        output_root.mkdir(parents=True, exist_ok=True)
        output_dir = Path(tempfile.mkdtemp(prefix="run_", dir=output_root))
        temp_dir = Path(tempfile.mkdtemp(prefix="temp_", dir=output_root))
        command = [
            *prefix, "align", *self._config_arguments(),
            "--output_format", "json", "-t", str(temp_dir),
            str(corpus_dir), self.dictionary, self.acoustic_model, str(output_dir),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"MFA corpus alignment failed (exit {result.returncode}): "
                f"{result.stderr[-1500:] or result.stdout[-1500:]}"
            )
        outputs = {path.stem: path for path in output_dir.rglob("*.json")}
        missing = [path.stem for path in transcripts if path.stem not in outputs]
        if missing:
            raise RuntimeError(f"MFA did not align {len(missing)} clips: {missing[:5]}")
        return {path.stem: parse_mfa_json(outputs[path.stem]) for path in transcripts}
