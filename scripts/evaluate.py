#!/usr/bin/env python
"""
Evaluates trained models and saves metrics.
"""
import os
import json
import torch
import argparse
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Audio
from jiwer import wer
import numpy as np
import torchaudio
import random


def evaluate_asr(manifest_path, model_dir, num_samples=5):
    """Evaluate ASR model on sample data"""
    # Load dataset
    ds = load_dataset("json", data_files={"test": manifest_path})["test"]

    # Load model and processor
    if os.path.exists(model_dir):
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_dir)
            model = Wav2Vec2ForCTC.from_pretrained(model_dir)
            print(f"Loaded ASR model from {model_dir}")
        except Exception as e:
            print(f"Could not load ASR model: {e}")
            # Use pretrained model as fallback for demo
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            print("Loaded pretrained ASR model as fallback")
    else:
        print(f"ASR model not found at {model_dir}, using pretrained model")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Sample a few examples
    if len(ds) > num_samples:
        examples = ds.select(random.sample(range(len(ds)), num_samples))
    else:
        examples = ds

    # Process audio files
    results = []
    for ex in examples:
        try:
            # Load audio file
            waveform, sample_rate = torchaudio.load(ex["file"])

            # Resample if needed
            if sample_rate != processor.feature_extractor.sampling_rate:
                waveform = torchaudio.transforms.Resample(sample_rate, processor.feature_extractor.sampling_rate)(
                    waveform
                )

            # Process audio
            input_values = processor(
                waveform.squeeze().numpy(),
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            ).input_values

            # Predict
            with torch.no_grad():
                logits = model(input_values).logits

            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]

            # Compare with ground truth
            results.append(
                {
                    "reference": ex["transcription"],
                    "prediction": transcription,
                }
            )

        except Exception as e:
            print(f"Error processing {ex['file']}: {e}")

    # Calculate WER
    if results:
        references = [r["reference"] for r in results]
        predictions = [r["prediction"] for r in results]
        metric = wer(references, predictions)
    else:
        metric = 1.0  # Maximum error if no results

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/asr_results.json", "w") as f:
        json.dump({"wer": metric, "samples": results}, f, indent=2)

    print(f"ASR WER: {metric:.4f}")
    return metric


def generate_sample_exercises(corpus_path, model_dir, num_samples=3):
    """Generate exercise examples using fine-tuned GPT-2"""
    # Load corpus
    with open(corpus_path, "r") as f:
        corpus = [line.strip() for line in f if line.strip()]

    # Load model and tokenizer
    if os.path.exists(model_dir):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            model = AutoModelForCausalLM.from_pretrained(model_dir)
            print(f"Loaded GPT-2 model from {model_dir}")
        except Exception as e:
            print(f"Could not load GPT-2 model: {e}")
            # Use pretrained model as fallback for demo
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = AutoModelForCausalLM.from_pretrained("gpt2")
            print("Loaded pretrained GPT-2 model as fallback")
    else:
        print(f"GPT-2 model not found at {model_dir}, using pretrained model")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")

    # Generate exercises
    exercises = []
    prompts = [
        "Complete the sentence: ",
        "Translate to English: ",
        "Fill in the blank: The ____ is an important cultural symbol.",
    ]

    for prompt in prompts[:num_samples]:
        try:
            # Sample a random sentence from corpus for context
            context = random.choice(corpus) if corpus else ""

            # Generate text
            input_text = f"{prompt}{context}"
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids

            outputs = model.generate(
                input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, early_stopping=True
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            exercises.append({"prompt": prompt, "context": context, "generated": generated_text})

        except Exception as e:
            print(f"Error generating text: {e}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/exercises.json", "w") as f:
        json.dump(exercises, f, indent=2)

    print(f"Generated {len(exercises)} sample exercises")
    return exercises


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/raw/toy_lang_dataset/manifest_train.jsonl")
    parser.add_argument("--corpus", default="data/raw/toy_lang_dataset/corpus.txt")
    parser.add_argument("--asr-model", default="models/asr")
    parser.add_argument("--gpt2-model", default="models/gpt2")
    args = parser.parse_args()

    print("\n=== Evaluating ASR Model ===")
    evaluate_asr(args.manifest, args.asr_model)

    print("\n=== Generating Sample Exercises ===")
    generate_sample_exercises(args.corpus, args.gpt2_model)

    print("\nEvaluation complete. Results saved to the 'results' directory.")


if __name__ == "__main__":
    main()
