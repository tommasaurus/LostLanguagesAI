#!/usr/bin/env python
"""
Downloads sample audio files from publicly available sources
to create a toy dataset for demonstration purposes.
"""
import os
import argparse
import urllib.request
import soundfile as sf
import torchaudio
import torch
import numpy as np
import json
import glob
from pathlib import Path
from tqdm import tqdm

# Common Voices samples or LibriSpeech samples can work as placeholder data
SAMPLE_URLS = [
    "https://github.com/mozilla/DeepSpeech/raw/master/data/smoke_test/smoke_test.wav",
    # You can add more URLs here from open audio datasets
]


def download_file(url, target_path):
    """Download a file from URL to target path"""
    print(f"Downloading {url} to {target_path}")
    urllib.request.urlretrieve(url, target_path)


def generate_synthetic_audio(text, output_path, duration=2.0, sample_rate=16000):
    """Generate a simple sine wave audio file as a placeholder"""
    print(f"Generating synthetic audio for: '{text}'")
    # Generate a simple tone
    frequencies = np.random.randint(200, 800, size=len(text.split()))
    audio = np.zeros(int(duration * sample_rate))

    # Create a simple audio waveform
    for i, freq in enumerate(frequencies):
        t = np.arange(0, 0.5, 1 / sample_rate)
        segment = 0.5 * np.sin(2 * np.pi * freq * t)
        offset = int(i * 0.5 * sample_rate)
        if offset + len(segment) <= len(audio):
            audio[offset : offset + len(segment)] = segment

    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Save as WAV
    sf.write(output_path, audio, sample_rate)
    return output_path


def convert_librispeech(librispeech_dir, output_dir, limit=20):
    """Convert LibriSpeech data to our format"""
    import glob
    from pathlib import Path
    import os
    import shutil

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio_16k"), exist_ok=True)

    # Find all transcript files
    transcript_files = glob.glob(f"{librispeech_dir}/*/*/*/*.txt")

    manifest = []
    transcripts = []
    count = 0

    # Process each transcript file
    for trans_file in transcript_files:
        if count >= limit:
            break

        with open(trans_file, "r") as f:
            for line in f:
                if count >= limit:
                    break

                parts = line.strip().split(" ", 1)
                if len(parts) != 2:
                    continue

                file_id, text = parts
                # Find corresponding flac file
                flac_path = os.path.join(os.path.dirname(trans_file), f"{file_id}.flac")

                if os.path.exists(flac_path):
                    # Copy and convert to WAV
                    wav_name = f"sample_{count+1}.wav"
                    wav_path = os.path.join(output_dir, "audio_16k", wav_name)

                    # Convert FLAC to WAV with torchaudio
                    waveform, sample_rate = torchaudio.load(flac_path)
                    torchaudio.save(wav_path, waveform, sample_rate)

                    # Create manifest entry
                    manifest.append(
                        {
                            "file": f"audio_16k/{wav_name}",
                            "transcription": text,
                            "speaker": f"librispeech_{file_id.split('-')[0]}",
                        }
                    )

                    # Create transcript entry
                    transcripts.append(f"{wav_name}\t{text}")

                    count += 1

    # Write manifest file
    with open(os.path.join(output_dir, "manifest_train.jsonl"), "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

    # Write transcripts file
    with open(os.path.join(output_dir, "transcripts.tsv"), "w") as f:
        for line in transcripts:
            f.write(line + "\n")

    print(f"Converted {count} LibriSpeech samples to {output_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/raw/toy_lang_dataset/audio_16k")
    ap.add_argument("--generate", action="store_true", help="Generate synthetic audio instead of downloading")
    ap.add_argument("--librispeech", help="Path to LibriSpeech directory to convert")
    ap.add_argument("--limit", type=int, default=20, help="Max samples to convert")
    args = ap.parse_args()

    # If LibriSpeech option is used, convert that data
    if args.librispeech:
        convert_librispeech(args.librispeech, Path(args.output_dir).parent, args.limit)
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Read transcriptions from the TSV file
    tsv_path = Path(args.output_dir).parent / "transcripts.tsv"
    if not tsv_path.exists():
        print(f"Transcription file {tsv_path} not found.")
        return

    transcriptions = []
    with open(tsv_path, "r") as f:
        for line in f:
            if line.strip():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    transcriptions.append((parts[0], parts[1]))

    if args.generate:
        # Generate synthetic audio for each transcription
        for i, (filename, text) in enumerate(tqdm(transcriptions)):
            output_path = os.path.join(args.output_dir, filename)
            generate_synthetic_audio(text, output_path)
    else:
        # Download real audio samples and rename them
        for i, url in enumerate(tqdm(SAMPLE_URLS)):
            if i < len(transcriptions):
                filename, _ = transcriptions[i]
                target_path = os.path.join(args.output_dir, filename)
                download_file(url, target_path)

        # For any remaining transcriptions, generate synthetic audio
        if len(transcriptions) > len(SAMPLE_URLS):
            for i in range(len(SAMPLE_URLS), len(transcriptions)):
                filename, text = transcriptions[i]
                output_path = os.path.join(args.output_dir, filename)
                generate_synthetic_audio(text, output_path)

    print(f"Created {len(transcriptions)} audio files in {args.output_dir}")


if __name__ == "__main__":
    import torch  # Import here so torch import time doesn't delay --help

    main()
