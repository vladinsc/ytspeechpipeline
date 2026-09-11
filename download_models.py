"""Pre-fetch every public model used by the pipeline into container caches."""

from __future__ import annotations

import json
import os
from pathlib import Path


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    hf_home = Path(os.environ.get("HF_HOME", "/opt/models/huggingface"))
    torch_home = Path(os.environ.get("TORCH_HOME", "/opt/models/torch"))
    nltk_home = Path(os.environ.get("NLTK_DATA", "/opt/models/nltk"))
    for directory in (hf_home, torch_home, nltk_home):
        directory.mkdir(parents=True, exist_ok=True)

    print("[models] Downloading faster-whisper large-v3 from Hugging Face...")
    from faster_whisper.utils import download_model

    whisper_path = download_model("large-v3")

    print("[models] Downloading Demucs htdemucs...")
    from demucs.pretrained import get_model

    demucs_model = get_model("htdemucs")
    del demucs_model

    print("[models] Downloading Silero VAD through Torch Hub...")
    import torch

    silero_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )
    del silero_model

    print("[models] Downloading the English Wav2Vec2 alignment model...")
    import torchaudio

    align_model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
    del align_model

    print("[models] Downloading NLTK English sentence-alignment data...")
    import nltk

    if not nltk.download("punkt_tab", download_dir=str(nltk_home), quiet=False):
        raise RuntimeError("NLTK punkt_tab download failed")

    manifest = {
        "whisper_model": "large-v3",
        "whisper_path": str(whisper_path),
        "demucs_model": "htdemucs",
        "vad_model": "silero_vad",
        "alignment_model": "WAV2VEC2_ASR_BASE_960H",
        "language": "en",
        "hf_cache_bytes": directory_size(hf_home),
        "torch_cache_bytes": directory_size(torch_home),
        "nltk_cache_bytes": directory_size(nltk_home),
    }
    manifest_path = Path("/opt/models/model-manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[models] Wrote {manifest_path}")


if __name__ == "__main__":
    main()
