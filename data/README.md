# Data Directory

This directory contains the data used for training and evaluating the models in the Lost Languages AI project.

## Directory Structure

- `raw/`: Contains the original, unprocessed data
  - `toy_lang_dataset/`: A small example dataset for demonstration purposes
    - `transcripts.tsv`: Tab-separated file with filename and transcription
    - `manifest_train.jsonl`: JSONL file with audio file paths and transcriptions
    - `corpus.txt`: Text corpus for language model training
    - `audio_16k/`: Directory containing 16kHz WAV audio files

## Data Acquisition for Real Projects

For a real endangered language preservation project, data can be obtained from:

1. **Linguistic archives**:
   - [The Language Archive](https://archive.mpi.nl/tla/)
   - [Endangered Languages Archive (ELAR)](https://www.elararchive.org/)
   - [California Language Archive](https://cla.berkeley.edu/)

2. **Field recordings**:
   - Collaborate with linguistic researchers
   - Partner with native speaker communities
   - Record native speakers with proper consent

3. **Existing documentation**:
   - Published dictionaries and grammars
   - Missionary texts
   - Anthropological records

## Data Preparation

Use the scripts in the `scripts/` directory to preprocess your data:

```bash
# Preprocess audio files
python scripts/preprocess_audio.py --data-dir data/raw/your_language --out-manifest data/raw/your_language/manifest_train.jsonl

# Once you have a manifest, you can train the ASR model
python scripts/train_asr.py --manifest data/raw/your_language/manifest_train.jsonl

# For vocabulary and exercise generation
python scripts/train_vocab.py --corpus data/raw/your_language/corpus.txt
python scripts/train_gpt2.py --corpus data/raw/your_language/corpus.txt
``` 