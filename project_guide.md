# Lost Languages AI: Proof of Concept Guide

This guide walks through running the complete proof of concept pipeline for the Lost Languages AI project.

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up directories:
```bash
mkdir -p models/asr models/vocab models/gpt2 results
```

## Running the Complete Pipeline

### 1. Preprocess Audio Data

First, preprocess raw audio files into a consistent format (16kHz mono WAV) and create a manifest file:

```bash
python scripts/preprocess_audio.py \
  --data-dir data/raw/toy_lang_dataset \
  --out-manifest data/raw/toy_lang_dataset/manifest_train.jsonl
```

### 2. Train Speech Recognition Model (wav2vec 2.0)

Fine-tune the wav2vec 2.0 model on your processed audio:

```bash
python scripts/train_asr.py \
  --manifest data/raw/toy_lang_dataset/manifest_train.jsonl \
  --out models/asr \
  --epochs 5
```

### 3. Mine Vocabulary (DistilBERT)

Fine-tune DistilBERT on your text corpus to create a language model:

```bash
python scripts/train_vocab.py \
  --corpus data/raw/toy_lang_dataset/corpus.txt \
  --out models/vocab \
  --epochs 5
```

### 4. Generate Learning Exercises (GPT-2)

Fine-tune GPT-2 to generate interactive exercises:

```bash
python scripts/train_gpt2.py \
  --corpus data/raw/toy_lang_dataset/corpus.txt \
  --out models/gpt2 \
  --epochs 3
```

### 5. Evaluate Results

View the evaluation metrics:

```bash
cat results/asr_metrics.json
cat results/vocab_perplexity.txt
cat results/gpt2_bleu.json
```

## Using Jupyter Notebooks

For a more interactive approach, you can use the Jupyter notebooks:

1. Start Jupyter:
```bash
jupyter notebook
```

2. Navigate to the `notebooks/` directory and open:
   - `01_wav2vec2_finetune.ipynb` for speech recognition
   - `02_distilbert_vocab.ipynb` for vocabulary mining
   - `03_gpt2_exercise_gen.ipynb` for exercise generation

## Getting More Data

For a real-world application, you would need more data. Here are some sources:

1. **Linguistic Archives**:
   - [ELAR](https://www.elararchive.org/)
   - [OLAC](http://www.language-archives.org/)
   - [PARADISEC](https://www.paradisec.org.au/)

2. **Field Work Collaboration**:
   - Partner with linguistics departments at universities
   - Contact organizations like [Living Tongues Institute](https://livingtongues.org/)
   - Reach out to community organizations of speakers of endangered languages

3. **Synthesized Data**:
   - For prototyping, you can create synthetic data using text-to-speech from related languages
   - Use data augmentation techniques: speed perturbation, adding background noise, pitch shifting

4. **Simulated Low-Resource Language**:
   - Consider using a subset of a well-resourced language to simulate the low-resource scenario
   - This provides a controlled environment for method validation

Remember that for actual endangered language preservation, ethical considerations around data collection and community involvement are paramount. 