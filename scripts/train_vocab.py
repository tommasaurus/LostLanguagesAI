#!/usr/bin/env python
"""
Fine‑tunes DistilBERT as a masked‑LM and logs perplexity.
"""
import argparse, json, evaluate
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="Plain‑text file")
    ap.add_argument("--out", default="models/vocab")
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    lines = [l.strip() for l in open(args.corpus) if l.strip()]
    ds = Dataset.from_dict({"text": lines}).train_test_split(test_size=0.1, seed=42)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True, remove_columns=["text"])

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    collator = DataCollatorForLanguageModeling(tok, mlm_probability=0.15)

    tr_args = TrainingArguments(
        args.out,
        per_device_train_batch_size=16,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_total_limit=1,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model, args=tr_args, train_dataset=ds["train"], eval_dataset=ds["test"], data_collator=collator
    )
    trainer.train()

    ppl_metric = evaluate.load("perplexity", module_type="metric")
    ppl = ppl_metric.compute(model_id=args.out, input_texts=lines[:100])["perplexity"]

    print(f"Perplexity: {ppl:.2f}")
    with open("results/vocab_perplexity.txt", "w") as f:
        f.write(f"{ppl:.2f}")


if __name__ == "__main__":
    main()
