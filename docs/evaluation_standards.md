# Evaluation Standards

This repo uses dataset-aware evaluation rather than one universal score for every task. The same model output can require different metrics depending on whether the task is multiple choice, short-answer QA, long-reference QA, truthfulness generation, RAG, or tool use.

## Metric Roles

Metrics are classified into four roles:

- **Primary**: used to decide pass/fail for a specific dataset profile.
- **Secondary**: reported for analysis but not usually used alone for pass/fail.
- **Diagnostic**: used to flag cases for review, especially possible over-credit or under-credit.
- **Judge**: LLM- or human-rubric based review used for ambiguous, open-ended, or high-stakes cases.

## Core Reference Metrics

| Metric | Role | Best For | Key Limitation |
|---|---|---|---|
| Exact Match | Primary | Short factual answers, dates, names, option labels | Too strict for paraphrases |
| Token F1 | Primary | Short-answer QA with minor wording variation | Lexical only |
| Contains Match | Diagnostic | Detecting embedded correct phrases | Can over-credit wrong long answers |
| Semantic Similarity | Secondary | Paraphrase-tolerant comparison | Current implementation is TF-IDF, not contextual embeddings |
| BLEU | Secondary | Translation-like generation | Weak primary metric for factual QA |
| ROUGE-L | Secondary/Primary for long references | Long answers and summary-style references | Lexical; can miss concise correct answers |
| LLM Judge Correctness | Judge | Ambiguous/open-ended correctness | Cost, variance, and judge bias |

## Scoring Profiles

| Profile | Primary Metrics | Recommended Datasets | Caveat |
|---|---|---|---|
| `short_answer_qa` | Exact Match, Token F1 | TriviaQA, SQuAD short answers, custom golden QA | Contains match is diagnostic only |
| `multiple_choice` | Exact Match | MMLU, ARC | Prefer exact option parsing when explanations are present |
| `long_reference_qa` | ROUGE-L, Semantic Similarity | Long-passage Natural Questions variants | Low scores can reflect reference-shape mismatch |
| `truthfulness_generation` | LLM judge correctness | TruthfulQA-style generation | Requires calibrated judge rubric |

## Dataset Inventory Principle

Every dataset should declare:

- `task_format`
- `answer_type`
- `reference_shape`
- `scoring_profile`
- primary and secondary metrics
- caveats

This metadata is stored in the dataset registry and copied into each normalized task so workflows can choose the right scoring profile automatically.

## Current Implementation Notes

- Semantic similarity currently uses deterministic local TF-IDF cosine similarity. This keeps demos offline and reproducible, but provider embeddings or BERTScore should be added as a stronger semantic option.
- BLEU and ROUGE are included for transparency and diagnostics. They should not be treated as universal quality scores.
- Natural Questions from `sentence-transformers/natural-questions` often provides long passages rather than concise gold answers. It is therefore assigned to `long_reference_qa` until a short-answer source/normalizer is added.
