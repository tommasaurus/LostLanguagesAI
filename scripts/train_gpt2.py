#!/usr/bin/env python
"""
Fine‑tunes GPT‑2 on prompt‑wrapped corpus, computes BLEU, saves model.
"""
import argparse, json, sacrebleu, torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)


def load_prompts(corpus):
    wrap = lambda s: f"<prompt>Dialogue:\n{s}\n<end>"
    return [wrap(l.strip()) for l in open(corpus) if l.strip()]


def bleu_score(model, tok, refs):
    preds = []
    for r in refs:
        inp = tok(r, return_tensors="pt").input_ids.to(model.device)
        gen = model.generate(inp, max_length=50, num_beams=5, do_sample=False)[0]
        preds.append(tok.decode(gen, skip_special_tokens=True).split())
    refs = [[r.split()] for r in refs]
    return sacrebleu.corpus_bleu(preds, refs).score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="models/gpt2")
    ap.add_argument("--model-name", default="gpt2")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    prompts = load_prompts(args.corpus)
    ds = Dataset.from_dict({"text": prompts}).train_test_split(test_size=0.1)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    tok.pad_token = tok.eos_token
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True, remove_columns=["text"])

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    tr_args = TrainingArguments(
        args.out,
        per_device_train_batch_size=8,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_total_limit=1,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model, args=tr_args, train_dataset=ds["train"], eval_dataset=ds["test"], data_collator=collator
    )
    trainer.train()

    bleu = bleu_score(model, tok, ds["test"]["input_ids"][:50])
    print("BLEU:", bleu)
    with open("results/gpt2_bleu.json", "w") as f:
        json.dump({"bleu": bleu}, f, indent=2)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
