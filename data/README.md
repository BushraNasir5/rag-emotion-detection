# Dataset Setup

## Required datasets

- GoEmotions (Hugging Face: `go_emotions`)
- EmpatheticDialogues (Hugging Face: `empathetic_dialogues`)
- SemEval-2018 Affect in Tweets (download manually due licensing/source package differences)

## Local paths used by this project

- `data/raw/semeval2018/train.csv`
- `data/raw/semeval2018/dev.csv`
- `data/raw/semeval2018/test.csv`
- `data/custom_eval/ea_rag_eval.jsonl`

## Reproducibility note

All experiment scripts can run with the included custom demo benchmark for smoke testing even before full datasets are downloaded.
