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
| Semantic Similarity | Secondary | Paraphrase-tolerant comparison | TF-IDF is the default; provider embeddings add cost/dependency when enabled |
| BLEU | Secondary | Translation-like generation | Weak primary metric for factual QA |
| ROUGE-L | Secondary/Primary for long references | Long answers and summary-style references | Lexical; can miss concise correct answers |
| LLM Judge Correctness | Judge | Ambiguous/open-ended correctness | Cost, variance, and judge bias |

## Scoring Profiles

| Profile | Primary Metrics | Primary Score Formula | Recommended Datasets | Caveat |
|---|---|---|---|---|
| `short_answer_qa` | Exact Match, Token F1 | `max(exact_match, 0.65 * token_f1 + 0.35 * semantic_similarity)` | TriviaQA, SQuAD short answers, custom golden QA | Contains match is diagnostic only |
| `multiple_choice` | Exact Match | exact match against answer text or displayed option label | MMLU, ARC | Prefer exact option parsing when explanations are present |
| `long_reference_qa` | ROUGE-L, Semantic Similarity | `max(0.55 * rouge_l + 0.45 * semantic_similarity, 0.50 * token_f1 + 0.50 * semantic_similarity)` | Long-passage Natural Questions variants | Low scores can reflect reference-shape mismatch |
| `truthfulness_generation` | LLM judge correctness | judge rubric score, with deterministic metrics as supporting evidence | TruthfulQA-style generation | Requires calibrated judge rubric |

## Dataset Inventory Principle

Every dataset should declare:

- `task_format`
- `answer_type`
- `reference_shape`
- `scoring_profile`
- primary and secondary metrics
- caveats

This metadata is stored in the dataset registry and copied into each normalized task so workflows can choose the right scoring profile automatically.

## Curated Dataset Principle

Curated repo-native datasets may combine compatible rows from public benchmark datasets, but curation must not rewrite the source question, expected answer, or reference answers. The curated layer may add only compatible standardized fields such as broad category, subcategory, source dataset, source task ID, source cache path, scoring profile, and integrity notes.

The first curated Answer Accuracy asset, `curated_knowledge_v1`, follows this policy:

- **Included sources**: compatible closed-book MMLU, TriviaQA, and ARC rows from local normalized benchmark caches.
- **Selection policy**: use the largest available compatible cache for each included source; do not sample during curation.
- **Excluded from curated v1**: the current Natural Questions cache, because its long-reference answer shape is better handled as a separate out-of-box benchmark profile.
- **Schema**: shared `EvalTask` JSONL fields plus provenance metadata.
- **Scoring**: source-specific scoring profiles are preserved at row level, then broad curated categories are reported for portfolio analysis. Notebook run modes sample at evaluation time to control model-call cost.
- **Default sampling**: source-first stratified sampling for curated benchmark runs. The sampler balances across source datasets first, then broad categories within each source, so report samples are easier to interpret than purely distribution-proportional random samples.

## Portfolio Reporting Principle

Multi-dataset runs are treated as an evaluation portfolio, not as one perfectly comparable score. Each dataset keeps its registered scoring profile and reference-shape caveats. Reports may show an overall directional summary, but dataset-level and category-level results remain the primary evidence.

The report separates two ratings:

- **Capability rating**: performance signal from pass rate and profile score.
- **Evaluation reliability**: confidence in the measurement, considering sample size, reference-shape fit, metric disagreement, judge rescues, and judge failures.

A low reliability rating does not automatically mean the model is weak. It means the evidence needs calibration or review before being used for model selection or governance conclusions.

## Current Implementation Notes

- Semantic similarity defaults to deterministic local TF-IDF cosine similarity. This keeps demos offline and reproducible. Provider embeddings can be enabled when `OPENAI_EMBEDDING_MODEL` is configured and the notebook/workflow opts into API embeddings.
- BLEU and ROUGE are included for transparency and diagnostics. They should not be treated as universal quality scores.
- Natural Questions from `sentence-transformers/natural-questions` often provides long passages rather than concise gold answers. It is therefore assigned to `long_reference_qa` until a short-answer source/normalizer is added.
