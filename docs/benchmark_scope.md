# EA-RAG Scope Lock

## Compute assumption

- Target profile: medium compute (single strong GPU or comparable cloud budget).
- Design choice: no full LLM fine-tuning; use modular adapters and controllable prompts.

## Final benchmark set

1. GoEmotions
   - Purpose: fine-grained emotion recognition validation and label mapping.
2. SemEval-2018 Task 1 (Affect in Tweets)
   - Purpose: robustness under domain shift.
3. EmpatheticDialogues
   - Purpose: emotional alignment and dialogue-style evaluation.
4. Custom EA-RAG benchmark (`data/custom_eval/ea_rag_eval.jsonl`)
   - Purpose: controlled experiments for retrieval faithfulness + tone adherence.
   - Fields:
     - `id`
     - `query`
     - `target_emotion`
     - `gold_context`
     - `reference_answer`
     - `tone_constraint`

## Model lock

- Emotion classifier: `j-hartmann/emotion-english-distilroberta-base`.
- Embedding model: `BAAI/bge-large-en-v1.5` with fallback `intfloat/e5-large-v2`.
- Generator:
  - Primary: external API model through adapter.
  - Reproducible fallback: local deterministic generator adapter.

## Experiment policy

- Baselines:
  - Vanilla RAG
  - Prompt-only emotion awareness
- Proposed:
  - Full EA-RAG
- Required ablations:
  - remove query rewrite
  - remove emotion-aware ranking
  - remove tone conditioning
  - sweep retrieval weights alpha, beta, gamma

## Reproducibility

- Config-driven runs through YAML.
- Seed control at script level.
- Output artifacts saved in `results/`.
