# GenAI Capability Bench

<div align="center">

**A modular benchmark suite for measuring what GenAI models and systems can do, and how well they do it**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Provider Agnostic](https://img.shields.io/badge/LLM-Provider%20Agnostic-412991)](.env.example)
[![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4?logo=microsoftazure)](.env.example)
[![OpenAI Compatible](https://img.shields.io/badge/OpenAI-Compatible-111827?logo=openai)](.env.example)
[![Notebooks](https://img.shields.io/badge/Demo-Notebooks-f37626?logo=jupyter)](notebooks/)

*Answer accuracy | truthfulness | instruction following | reasoning | RAG capability | tool use | agentic task completion*

</div>

---

## 🎯 Why This Repo

Most LLM evaluation projects quickly collapse into one of two shapes:

1. benchmark scripts that answer only *"which model got the most answers right?"*
2. safety tests that answer mainly *"where can the model go wrong?"*

This repo is focused on a different question:

> **What GenAI capabilities does a model or system have, and how reliably can it perform them across task types?**

The goal is to build a reusable capability testing suite that can evaluate both
base LLM behavior and system-level GenAI workflows. Safety and risk signals may
be captured as supporting diagnostics, but the organizing principle is
**capability measurement**, not red-team testing.

---

## 🧭 At a Glance

```mermaid
flowchart LR
    A["Datasets and Tasks"] --> B["Capability Evaluators"]
    C["Model Providers"] --> B
    B --> D["Normalized Results"]
    D --> E["Scorecards"]
    D --> F["Leaderboards"]
    D --> G["Reports and Notebooks"]

    subgraph Capabilities
        B1["Answer Accuracy"]
        B2["Truthfulness"]
        B3["Instruction Following"]
        B4["Reasoning and Logic"]
        B5["RAG Capability"]
        B6["Tool Use"]
        B7["Agentic Task Completion"]
    end

    B --> B1
    B --> B2
    B --> B3
    B --> B4
    B --> B5
    B --> B6
    B --> B7
```

Every evaluator emits the same result contract:

```text
run_id + model + capability + task + output + metrics + score + pass/fail + cost + latency + metadata
```

That shared schema is the backbone of the project: it lets answer accuracy,
TruthfulQA-style truthfulness, instruction following, reasoning, RAG, tool use,
and agent workflows all roll up into comparable scorecards.

---

## 🧩 Capability Taxonomy

| Capability | What It Measures | Starter Metrics |
|---|---|---|
| **Answer Accuracy** | Whether the model answers factual questions correctly across domains such as history, science, math, law, literature, and business topics | exact match, contains match, token F1, semantic similarity |
| **Truthfulness** | Whether the model avoids common misconceptions and aligns more closely with correct answers than known incorrect answers | correct-vs-incorrect similarity margin, future LLM judge rubric |
| **Instruction Following** | Whether the model follows explicit user constraints | JSON validity, required terms, forbidden terms, word limits, regex checks |
| **Reasoning and Logic** | Whether the model can derive correct answers through arithmetic, symbolic reasoning, multi-hop logic, and contradiction handling | final-answer matching, future stepwise reasoning checks |
| **RAG Capability** | Whether a RAG system retrieves useful context and generates grounded, complete, cited answers | retrieval hit rate, groundedness, completeness, faithfulness, citation support |
| **Tool Use** | Whether the model chooses the right tools and calls them correctly | tool precision/recall/F1, argument validity, sequence quality |
| **Agentic Task Completion** | Whether an agent can plan, execute, validate, and complete tasks under cost and latency constraints | task score, trace completeness, tool correctness, cost, latency |

The first implementation focuses on the **core LLM capabilities**. RAG, tool use,
and agentic task completion are planned as adapters that align with existing local
evaluation repos rather than duplicating them.

---

## 🛠️ Current Build

The repo currently includes a working vertical slice:

```text
Core LLM Capability MVP
|-- answer accuracy evaluator
|-- truthfulness evaluator
|-- instruction-following evaluator
|-- reasoning / logic evaluator
|-- provider-agnostic model client interface
|-- deterministic mock client for smoke tests
|-- normalized result schema
|-- CLI runner
|-- summary tables and plots
`-- five demo notebooks
```

The smoke demo runs without API credentials by using a deterministic mock model.
Real provider runs can use OpenAI-compatible or Azure OpenAI settings through
environment variables.

---

## 📓 Demo Notebooks

The notebooks are designed as user-friendly demos and analysis runbooks. Heavy
logic lives in `src/`; notebooks stay readable, interactive, and report-oriented.

| Notebook | Purpose |
|---|---|
| [`01_answer_accuracy.ipynb`](notebooks/01_answer_accuracy.ipynb) | Evaluate factual QA and slice results by domain/category |
| [`02_truthfulness.ipynb`](notebooks/02_truthfulness.ipynb) | Run TruthfulQA-style correct-vs-incorrect scoring |
| [`03_instruction_following.ipynb`](notebooks/03_instruction_following.ipynb) | Check JSON, formatting, length, and constraint adherence |
| [`04_reasoning_logic.ipynb`](notebooks/04_reasoning_logic.ipynb) | Evaluate final-answer correctness for reasoning tasks |
| [`05_core_llm_leaderboard.ipynb`](notebooks/05_core_llm_leaderboard.ipynb) | Aggregate core capability results into a model scorecard |

Notebook flow:

```text
Capability overview
  -> dataset/task selection
  -> model selection
  -> evaluation run
  -> score summary
  -> visualization
  -> executive summary
  -> saved artifacts
```

---

## 🚀 Quickstart

Clone and install:

```bash
git clone https://github.com/minw0607/genai_capability_bench.git
cd genai_capability_bench

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Run the local smoke demo:

```bash
python -m genai_capability_bench.core.runner configs/eval_core_demo.yaml
```

Expected artifact layout:

```text
outputs/runs/core_demo/
|-- metadata.json
|-- results.csv
|-- results.json
`-- summary.csv
```

Example summary from the smoke demo:

| Model | Capability | Category | Avg Score | Pass Rate |
|---|---|---|---:|---:|
| DemoMock | answer_accuracy | history | 1.00 | 1.00 |
| DemoMock | answer_accuracy | science | 1.00 | 1.00 |
| DemoMock | instruction_following | structured_output | 1.00 | 1.00 |
| DemoMock | reasoning_logic | arithmetic | 1.00 | 1.00 |
| DemoMock | truthfulness | general_misconceptions | 0.79 | 1.00 |

---

## 🔌 Provider Setup

Provider settings are read from `.env`:

```bash
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_VERSION=

OPENAI_APIM_HEADER_NAME=
OPENAI_APIM_SUBSCRIPTION_KEY=

OPENAI_GENERATION_MODEL=gpt-4o
OPENAI_JUDGE_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

For Azure OpenAI, set `OPENAI_API_VERSION`. For direct OpenAI-compatible endpoints,
leave it blank and set `OPENAI_BASE_URL` as needed.

---

## ⚙️ Configuration-Driven Runs

Evaluations are configured with YAML:

```yaml
run_id: core_demo
dataset: datasets/samples/core_demo_tasks.json
output_dir: outputs/runs
default_pass_threshold: 0.7

models:
  - name: DemoMock
    provider: mock
    model: mock-model
```

Task files can be JSON, JSONL, or CSV. Each task maps into the shared
`EvalTask` schema:

```json
{
  "task_id": "accuracy_history_001",
  "capability": "answer_accuracy",
  "category": "history",
  "input_text": "Who was the first president of the United States?",
  "expected_output": "George Washington",
  "references": ["George Washington"]
}
```

---

## 🗂️ Repository Layout

```text
configs/                         Evaluation and model configs
datasets/samples/                 Small local demo datasets
docs/                             Design notes and development plan
notebooks/                        Demo-style capability notebooks
outputs/runs/                     Generated run artifacts
src/genai_capability_bench/
  adapters/                       Future RAG and Agent integration adapters
  capabilities/                   Capability-specific evaluators
  clients/                        Model client abstractions
  core/                           Schemas, experiment runner, registry
  metrics/                        Reusable scoring functions
  reporting/                      Tables, plots, summaries
tests/                            Unit and smoke tests
```

---

## 🛣️ Integration Roadmap

This repo is intended to sit above specialized evaluation repos and normalize
their outputs into capability scorecards.

```mermaid
flowchart TB
    A["GenAI Capability Bench"] --> B["Core LLM Evaluators"]
    A --> C["RAG Adapter"]
    A --> D["Agent Adapter"]

    C --> E["Existing RAG Evaluation Framework"]
    D --> F["Existing Agent / Multi-Agent Framework"]

    E --> G["Groundedness, completeness, faithfulness, citations"]
    F --> H["Task completion, tool correctness, traces, cost"]
```

Planned phases:

| Phase | Focus | Status |
|---|---|---|
| 1 | Core LLM MVP: answer accuracy, truthfulness, instruction following, reasoning | Started |
| 2 | Stronger semantic metrics and LLM-as-judge rubric library | Planned |
| 3 | Richer benchmark datasets and domain slicing | Planned |
| 4 | RAG adapter aligned with the existing RAG evaluation framework | Planned |
| 5 | Tool-use and agent adapters aligned with the existing Agent repo | Planned |
| 6 | Cross-run regression reporting and model comparison dashboards | Planned |

---

## 🛡️ Relationship to Safety Evaluation

Safety, robustness, and misuse testing are important, but this project is not
primarily a safety red-team suite. The capability lens asks:

- Can the model answer correctly?
- Can it reason?
- Can it follow constraints?
- Can it use context?
- Can it call tools?
- Can it complete agentic tasks?

Safety-related outcomes may be captured as metadata or diagnostic signals where
they affect capability, but they do not define the top-level taxonomy.

---

## 🧪 Development Notes

- Keep reusable implementation in `src/`.
- Keep notebooks coding-light and analysis-heavy.
- Make every run reproducible through config files.
- Preserve raw outputs and metric details for auditability.
- Prefer adapters over duplication when integrating with RAG or Agent repos.

See:

- [`docs/development_plan.md`](docs/development_plan.md)
- [`docs/capability_taxonomy.md`](docs/capability_taxonomy.md)
- [`docs/repo_integration_strategy.md`](docs/repo_integration_strategy.md)
