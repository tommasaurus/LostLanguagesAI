# Saving Endangered Languages with AI : Proof of Concept  
Tommy Qu · Ronish Dua · Abhaya Upraity  

---

## Why this matters
> **40 % of the world’s 7 000 languages are endangered**—many with < 1 000 speakers.  
> When a language disappears, so do centuries of oral history, ecological knowledge, and cultural identity (UNESCO 2010).

Traditional documentation—linguists recording, transcribing, and publishing grammars—cannot keep pace with the rate of language loss.  
Our project shows that **modern self‑supervised and few‑shot learning techniques can bootstrap usable resources from only a few hours of audio and a small text corpus**, paving the way for community‑owned revitalization tools.

---

## Project goals
1. **Automate transcription** of raw speech in an endangered tongue using transfer‑learned **wav2vec 2.0**.  
2. **Mine vocabulary** and generate bilingual word lists via a few‑shot **DistilBERT** masked‑LM.  
3. **Produce culturally relevant learning material**—dialogues, fill‑in‑the‑blank exercises—by prompt‑tuning **GPT‑2**.  
4. Evaluate accuracy (WER, perplexity, BLEU), document strengths + limitations, and publish an open, reproducible pipeline.

---

## Methodology

| Stage | Model | Training recipe |
|-------|-------|-----------------|
| **Speech Recognition** | `facebook/wav2vec2-large-xlsr-53` | 10 h labelled audio → speed + noise augmentation → fine‑tune 5 epochs, LR 1e‑4 |
| **Vocabulary mining** | `distilbert-base-uncased` | 100–500 sentences → masked‑LM fine‑tune, perplexity target < 25 |
| **Exercise generation** | `gpt2` (small) | Same corpus with instructional prompts → beam search 5, early stopping |

---