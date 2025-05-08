#!/usr/bin/env python
"""
Fine‑tunes wav2vec 2.0 on a JSONL manifest, logs WER,
and saves processor + model.
"""
import argparse, json, os, torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from jiwer import wer


def jsonl_dataset(path, root, sr=16_000):
    ds = load_dataset("json", data_files={"train": path})["train"]
    ds = ds.cast_column(
        "file",
        Audio(
            sampling_rate=sr, decode=True, stored_as_path=True, download_config=None, base_path=os.path.dirname(root)
        ),
    )
    ds = ds.train_test_split(test_size=0.1, seed=42)
    return ds


def prepare(ds, processor, sr):
    def _map(batch):
        audio = batch["file"]["array"]
        batch["input_values"] = processor(audio, sampling_rate=sr).input_values[0]
        with processor.as_target_processor():
            batch["labels"] = processor(batch["transcription"]).input_ids
        return batch

    return ds.map(_map, remove_columns=ds["train"].column_names, num_proc=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="models/asr")
    ap.add_argument("--model-name", default="facebook/wav2vec2-base-960h")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    sr = processor.feature_extractor.sampling_rate
    root = os.path.dirname(args.manifest)
    ds = jsonl_dataset(args.manifest, root, sr)
    ds = prepare(ds, processor, sr)

    model = Wav2Vec2ForCTC.from_pretrained(args.model_name, vocab_size=len(processor.tokenizer))

    args_tr = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        num_train_epochs=args.epochs,
        fp16=torch.cuda.is_available(),
        save_total_limit=1,
        logging_steps=20,
    )

    def metrics(eval_pred):
        pred_ids = torch.argmax(torch.tensor(eval_pred.predictions), dim=-1)
        preds = processor.batch_decode(pred_ids)
        labels = processor.batch_decode(eval_pred.label_ids, group_tokens=False)
        return {"wer": wer(labels, preds)}

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=processor.feature_extractor,
        compute_metrics=metrics,
    )
    trainer.train()
    results = trainer.evaluate()
    print(results)

    os.makedirs("results", exist_ok=True)
    with open("results/asr_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    model.save_pretrained(args.out)
    processor.save_pretrained(args.out)


if __name__ == "__main__":
    main()
