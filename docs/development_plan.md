# Development Plan

## Mission

Build a modular benchmark suite for evaluating GenAI capabilities: what models
and systems can do, how well they do it, and where capability differences appear
across task families.

## Scope Boundary

This repository is capability-first. Safety signals may be captured as metadata,
but safety red-teaming is not the central taxonomy.

Existing local repos are integration targets:

- `/Users/minwu/Documents/GenAI/RAG/Repo/llm-eval-framework`
- `/Users/minwu/Documents/GenAI/Agent`

The capability suite should align with their metric definitions and consume their
outputs where useful, without duplicating their full RAG or agent systems.

## Phases

1. Core LLM capability MVP: answer accuracy, truthfulness, instruction following,
   reasoning and logic.
2. Shared run schema, model client layer, CLI runner, and reporting helpers.
3. Demo notebooks that call into `src/`.
4. RAG adapter aligned with the existing RAG framework metrics.
5. Tool-use and agentic task completion adapters aligned with the existing Agent repo.
6. Cross-capability leaderboard and regression reporting.

