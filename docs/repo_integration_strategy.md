# Integration Strategy

## RAG Repo

Use the RAG repo as the source of truth for full RAG pipeline evaluation. This
repo should align with or wrap stable metric concepts such as exact match, F1,
groundedness, completeness cascade, faithfulness, citation detection, retrieval
hit rate, cost tracking, and low-quality diagnosis.

## Agent Repo

Use the Agent repo as the source of truth for agent execution, tracing, and
single-vs-multi-agent comparisons. This repo should align with or wrap tool
precision/recall/F1, order accuracy, task completion, trace completeness, cost,
and latency metrics.

## Boundary

Adapters should ingest outputs or call stable APIs. They should not copy entire
pipeline implementations into this repo.

