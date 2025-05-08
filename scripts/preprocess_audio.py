#!/usr/bin/env python
"""
Resamples audio to 16 kHz mono WAV and writes a HuggingFace‑style
manifest (JSONL):
{"file": "audio/abc.wav", "transcription": "text ...", "speaker": "id"}
"""
import argparse, json, os, shutil, soundfile as sf, torchaudio
from pathlib import Path
from tqdm import tqdm


def resample(in_path: Path, out_path: Path, target_sr=16_000):
    audio, sr = sf.read(in_path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(torch.tensor(audio).T, sr, target_sr).T.numpy()
    sf.write(out_path, audio, target_sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder containing WAV/MP3 + a transcripts.tsv (file<TAB>text)")
    ap.add_argument("--out-manifest", required=True, help="Output JSONL path")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    audio_out = data_dir / "audio_16k"
    audio_out.mkdir(exist_ok=True)

    # Expect a tab‑separated transcripts file: filename<TAB>text
    tsv = data_dir / "transcripts.tsv"
    lines = [l.strip().split("\t") for l in tsv.open()]
    manifest = []

    for fname, text in tqdm(lines):
        src = data_dir / fname
        dst = audio_out / (Path(fname).stem + ".wav")
        resample(src, dst)
        manifest.append({"file": str(dst.relative_to(data_dir)), "transcription": text, "speaker": "unknown"})

    with open(args.out_manifest, "w") as f:
        for m in manifest:
            f.write(json.dumps(m) + "\n")


if __name__ == "__main__":
    import torch  # after argparse so torch import time doesn’t delay --help

    main()
