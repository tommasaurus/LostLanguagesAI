#!/usr/bin/env python
"""
Downloads and converts VoxPopuli dataset from Hugging Face to our format.
VoxPopuli contains European Parliament speech recordings in 18 languages.
"""
import os
import argparse
import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def convert_voxpopuli(language="cs", output_dir="data/raw/voxpopuli_dataset", limit=20):
    """Convert VoxPopuli data to our format"""
    print(f"Downloading VoxPopuli {language} dataset...")
    ds = load_dataset("facebook/voxpopuli", language)

    print(f"Dataset loaded with {len(ds['train'])} training examples")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "audio_16k"), exist_ok=True)

    # Select a subset of examples if limit is specified
    if limit > 0 and len(ds["train"]) > limit:
        # Select examples with different speakers for diversity
        speaker_ids = list(set(ds["train"]["speaker_id"]))
        selected_examples = []

        for speaker in speaker_ids[: min(limit, len(speaker_ids))]:
            examples = [i for i, ex in enumerate(ds["train"]) if ex["speaker_id"] == speaker]
            if examples:
                selected_examples.append(examples[0])
                if len(selected_examples) >= limit:
                    break

        # If we don't have enough different speakers, add more examples
        if len(selected_examples) < limit:
            remaining = limit - len(selected_examples)
            available = [i for i in range(len(ds["train"])) if i not in selected_examples]
            if available:
                selected_examples.extend(available[:remaining])

        examples = ds["train"].select(selected_examples)
    else:
        examples = ds["train"].select(range(min(limit, len(ds["train"]))))

    manifest = []
    transcripts = []

    # Process each example
    for i, example in enumerate(tqdm(examples, desc="Converting examples")):
        try:
            # Extract audio and save as WAV
            wav_name = f"sample_{i+1}.wav"
            wav_path = os.path.join(output_dir, "audio_16k", wav_name)

            # Save audio to WAV file
            sample_rate = example["audio"]["sampling_rate"]
            waveform = torch.tensor(example["audio"]["array"]).unsqueeze(0)
            torchaudio.save(wav_path, waveform, sample_rate)

            # Get transcription
            text = example["normalized_text"]

            # Create manifest entry
            manifest.append(
                {
                    "file": f"audio_16k/{wav_name}",
                    "transcription": text,
                    "speaker": f"voxpopuli_{example['speaker_id']}",
                }
            )

            # Create transcript entry
            transcripts.append(f"{wav_name}\t{text}")

        except Exception as e:
            print(f"Error processing example {i}: {e}")

    # Write manifest file
    manifest_path = os.path.join(output_dir, "manifest_train.jsonl")
    with open(manifest_path, "w") as f:
        for item in manifest:
            f.write(json.dumps(item) + "\n")

    # Write transcripts file
    transcripts_path = os.path.join(output_dir, "transcripts.tsv")
    with open(transcripts_path, "w") as f:
        for line in transcripts:
            f.write(line + "\n")

    # Create corpus file from transcriptions
    corpus_path = os.path.join(output_dir, "corpus.txt")
    with open(corpus_path, "w") as f:
        for item in manifest:
            f.write(item["transcription"] + "\n")

    print(f"Converted {len(manifest)} VoxPopuli samples to {output_dir}")
    print(f"Files created:")
    print(f"  - {manifest_path}")
    print(f"  - {transcripts_path}")
    print(f"  - {corpus_path}")

    return len(manifest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="cs", help="Language code to download (e.g., cs, de, fr)")
    parser.add_argument("--output-dir", default="data/raw/voxpopuli_dataset")
    parser.add_argument("--limit", type=int, default=20, help="Max samples to convert")
    args = parser.parse_args()

    convert_voxpopuli(args.language, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
