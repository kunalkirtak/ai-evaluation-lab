# RAG Evaluation & Benchmarking System

A compact, dependency-light framework for **scientifically evaluating a Retrieval-Augmented Generation (RAG) pipeline** — not another RAG chatbot demo.

It measures retrieval quality, answer accuracy, faithfulness, hallucination, latency, token usage, and cost, and produces CSV/JSON results, plots, an experiment comparison, and an error analysis, all from a single command.

---

## Why This Project Exists

Most public RAG portfolio projects stop at "build a chatbot that answers questions from my documents." That demonstrates plumbing, not judgment. What's usually missing is the harder, more valuable skill: **knowing whether the system is actually any good, and why it fails when it does.**

This project exists to demonstrate that skill directly:

- Design of retrieval and generation metrics that measure genuinely different things.
- Awareness that **retrieval quality ≠ answer quality**, and **faithfulness ≠ correctness**.
- Structured experimentation (comparing `top_k` configurations) rather than a single unverified run.
- Systematic error analysis instead of eyeballing a few outputs.
- Engineering discipline: type hints, dataclasses, logging, tests, and a pipeline that degrades gracefully instead of crashing when optional dependencies (like an LLM API) are unavailable.

## Problem Statement

Given a small local knowledge base and a set of evaluation questions with known ground-truth answers and known relevant documents, build a pipeline that:

1. Retrieves relevant context for each question.
2. Generates an answer from that context.
3. Scores the *retrieval* and the *generation* independently, using both deterministic, rule-based metrics and embedding-based semantic metrics.
4. Reports results in a form a reviewer can actually use: numbers, plots, and a per-example breakdown of what went wrong.

---

## Architecture

```
Documents
   │
   ▼
Document Loader        (src/data_loader.py)
   │
   ▼
Chunking                (src/chunker.py)
   │
   ▼
Embedding                (src/embeddings.py)
   │
   ▼
Retriever               (src/retriever.py)
   │
   ▼
Top-K Context
   │
   ▼
Answer Generator        (src/generator.py)
   │
   ▼
Evaluation Engine       (src/evaluator.py)
   │
   ▼
Metrics                 (src/metrics.py, src/llm_judge.py)
   │
   ▼
Evaluation Report        (src/reporting.py)
   │
   ▼
Visualizations           (results/plots/*.png)
```

Each stage is a small, independently testable module. There is no orchestration framework (no LangChain) because the pipeline is linear and simple enough that a framework would add indirection without adding value.

---

## Evaluation Dimensions

| Dimension | What it answers |
|---|---|
| Retrieval Quality | Did we find the right supporting documents at all? |
| Answer Accuracy | Does the generated answer match the ground truth meaning? |
| Faithfulness | Is the answer actually supported by what was retrieved? |
| Answer Relevance | Does the answer address the question that was asked? |
| Hallucination | Proxy for unsupported content in the answer (`1 - faithfulness`). |
| Latency | How long does retrieval / generation / evaluation take? |
| Token Usage | Rough estimate of input/output tokens consumed. |
| Cost | Configurable cost estimate derived from token usage. |

### Important conceptual distinction

**Retrieval quality ≠ answer quality.** A retriever can return exactly the right documents and the generator can still produce a poor, off-target, or inaccurate answer. Conversely, a weak retriever can sometimes get lucky and the generator can still produce a reasonable answer from partial context.

**Faithfulness ≠ correctness.** An answer can be entirely faithful to the retrieved context (every claim traces back to something in the context) while still failing to actually answer the question — for example, if the retriever pulled the wrong document, a faithful summary of that wrong document is still a wrong answer. This project measures both dimensions *separately and explicitly* rather than collapsing them into a single "quality" score.

---

## Metrics

### Deterministic metrics (retrieval)

These are exact, rule-based computations over document-ID overlap. They require no model and are fully reproducible.

| Metric | Formula |
|---|---|
| Precision@K | `relevant retrieved documents / retrieved documents` |
| Recall@K | `relevant retrieved documents / total relevant documents` |
| MRR | Reciprocal rank of the first relevant document retrieved |
| Hit Rate@K | 1.0 if at least one relevant document was retrieved, else 0.0 |

Implemented in `src/metrics.py`, unit-tested in `tests/test_metrics.py`.

### Model-based metrics (generation)

These rely on `sentence-transformers` (`all-MiniLM-L6-v2`) cosine similarity as a **proxy** for semantic agreement, because exact-string comparison is meaningless for free-text answers ("Paris is the capital of France" and "France's capital city is Paris" should score as equivalent, not different).

| Metric | Method |
|---|---|
| Answer Accuracy | Cosine similarity between generated answer and ground truth |
| Answer Relevance | Cosine similarity between question and generated answer |
| Faithfulness | Per-sentence: split the answer into sentences, embed each, take its max similarity to any retrieved context chunk, and compute the fraction of sentences above a support threshold |
| Hallucination Rate | `1 - faithfulness_score` |

### Why deterministic vs. model-based methodology was chosen per metric

- Retrieval metrics use **document-ID overlap** because relevance in this project is defined by a human-labeled ground-truth mapping (`relevant_document_ids`), so the correct metric is a precise set-overlap calculation, not a similarity estimate.
- Generation metrics use **embedding similarity** because there is no fixed vocabulary of "correct" free-text answers — semantic equivalence, not string equivalence, is the right notion of correctness, and embeddings are the lightweight, dependency-friendly way to approximate that.

### Limitations of embedding-based evaluation

- Cosine similarity of sentence embeddings is a **proxy**, not ground truth. Two sentences can be superficially similar in wording but differ in an important fact (a number, a negation, a name), and the embedding model may not penalize that difference enough.
- The faithfulness metric checks *semantic support*, not logical entailment — it cannot fully distinguish "this claim is implied by the context" from "this claim merely uses similar words to the context."
- `hallucination_rate = 1 - faithfulness_score` is explicitly a **project-level proxy**, not a validated hallucination-detection system. It should be read as "how much of the answer failed the semantic-support check," not as a guarantee of factual correctness or incorrectness.

### Optional LLM-as-judge

If `USE_LLM_JUDGE=true` and `GOOGLE_API_KEY` is set, `src/llm_judge.py` additionally asks Gemini to score correctness, faithfulness, and relevance and returns strict, validated JSON. If the API call fails, the response can't be parsed, or the JSON doesn't validate, the judge result is marked `available=False` and logged — **the deterministic metrics remain the source of truth** and the pipeline continues uninterrupted. LLM-as-judge scores also have known limitations: judge models can be inconsistent across runs, are sensitive to prompt phrasing, and can share blind spots with the model being evaluated.

---

## Retrieval Methodology

1. Documents in `data/documents/*.txt` are loaded (`src/data_loader.py`).
2. Each document is split into overlapping word-window chunks (`src/chunker.py`, default 60 words with 15-word overlap; short documents become a single chunk).
3. Every chunk is embedded with `sentence-transformers/all-MiniLM-L6-v2` (`src/embeddings.py`).
4. At query time, the question is embedded and compared against every chunk embedding via cosine similarity; the top-k chunks are returned with their scores and ranks (`src/retriever.py`).

If `sentence-transformers` model weights cannot be downloaded (e.g. no internet access), `EmbeddingModel` transparently falls back to a deterministic hashing-based bag-of-words embedding so the pipeline still runs end-to-end rather than crashing. This fallback is logged clearly and exists purely for robustness in constrained environments — for meaningful evaluation numbers, real network access to download the sentence-transformers model is expected.

## Generation Methodology

Two modes, selected by configuration:

- **MODE 1 — Local/deterministic (default):** the generator splits retrieved context into sentences, embeds them alongside the question, and stitches together the most question-relevant sentences into an answer. No network access or API key required. This is what the default demo uses.
- **MODE 2 — Optional LLM:** if `USE_LLM_GENERATION=true` and `GOOGLE_API_KEY` is set, Gemini generates the final answer from the retrieved context instead. No API call is ever made unless this is explicitly enabled, and no API call happens at import time. If the call fails for any reason, the generator logs the failure and transparently falls back to Mode 1.

---

## Experiment Design

The project runs **three configurations** over the same 15-question evaluation set:

| Experiment | `top_k` |
|---|---|
| Primary | 3 |
| Experiment A | 2 |
| Experiment B | 4 |

Results for Experiment A and B are aggregated into `results/experiment_comparison.csv`, showing how retrieval depth trades off against retrieval quality, answer quality, faithfulness, and latency — e.g. a larger `top_k` typically raises recall/hit-rate but can dilute faithfulness if the extra chunks are less relevant, and always increases latency and token usage.

---

## Error Analysis

Every evaluated example is classified into one failure category (`src/evaluator.py::classify_failure`), checked in priority order:

1. `retrieval_failure` — no relevant document was retrieved at all.
2. `unsupported_answer` — faithfulness score fell below the support threshold.
3. `low_relevance` — the answer doesn't semantically address the question.
4. `generation_failure` — retrieval succeeded but answer accuracy is still low.
5. `correct_answer` — none of the above triggered.

All examples (not just failures) are written to `results/error_analysis.csv` with the question, expected answer, generated answer, retrieved documents, and every relevant score, so successes and failures can be compared side by side.

---

## Project Structure

```
rag-evaluation-benchmark/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── colab_setup.py
├── colab_run.py
├── run_evaluation.py
├── generate_report.py
│
├── src/
│   ├── __init__.py
│   ├── config.py          # Config dataclass, paths, env-driven flags
│   ├── logging_config.py  # Centralized logging setup
│   ├── models.py           # Typed dataclasses shared across the pipeline
│   ├── data_loader.py      # Load documents + evaluation dataset
│   ├── chunker.py          # Word-window chunking with overlap
│   ├── embeddings.py       # sentence-transformers wrapper + offline fallback
│   ├── retriever.py        # Cosine-similarity top-k retriever
│   ├── generator.py        # Local extractive + optional Gemini generation
│   ├── metrics.py          # Precision/Recall/MRR/Hit-Rate + semantic metrics
│   ├── evaluator.py        # EvaluationEngine, per-question orchestration
│   ├── llm_judge.py        # Optional Gemini LLM-as-judge
│   └── reporting.py        # Summary stats, terminal report, plots, CSVs
│
├── data/
│   ├── documents/           # 10 short knowledge-base documents (.txt)
│   └── evaluation_dataset.json  # 15 labeled evaluation questions
│
├── results/                 # Generated: CSVs, JSON summary, plots/
│   └── plots/
│
└── tests/
    ├── test_metrics.py
    ├── test_retriever.py
    └── test_evaluator.py
```

---

## Installation

Requires Python 3.10+.

```bash
git clone <your-fork-url>
cd rag-evaluation-benchmark
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No API key is required for the default run.

## Google Colab Instructions

1. Open a new Colab notebook.
2. Upload `rag-evaluation-benchmark.zip` (or clone the repo) and extract it:
   ```python
   !unzip -q rag-evaluation-benchmark.zip -d /content/
   %cd /content/rag-evaluation-benchmark
   ```
3. Run the setup helper (creates any missing directories, installs requirements):
   ```python
   !python colab_setup.py
   ```
4. Run the full demo in one command:
   ```python
   !python colab_run.py
   ```
5. Run the tests:
   ```python
   !python -m pytest tests/ -v
   ```
6. Download results (e.g. via the Colab file browser, or):
   ```python
   from google.colab import files
   files.download("/content/rag-evaluation-benchmark/results/evaluation_summary.json")
   ```
7. Upload the project to GitHub (from Colab or locally):
   ```bash
   git init
   git add .
   git commit -m "RAG evaluation & benchmarking system"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

## VS Code Instructions

1. Open the `rag-evaluation-benchmark/` folder in VS Code.
2. Select/create a Python 3.10+ interpreter (Command Palette → "Python: Select Interpreter").
3. Open an integrated terminal and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the pipeline or tests directly from the terminal (see below), or use VS Code's built-in test explorer (pytest) to run/debug individual tests.

---

## Running the Project

```bash
python run_evaluation.py
```

This loads the documents and evaluation dataset, builds embeddings, runs the primary experiment (`top_k=3`) plus Experiment A (`top_k=2`) and Experiment B (`top_k=4`), and writes:

- `results/evaluation_results.csv`
- `results/evaluation_summary.json`
- `results/experiment_comparison.csv`
- `results/error_analysis.csv`
- `results/plots/*.png`

and prints a terminal report for the primary experiment.

To regenerate the terminal report and plots from an existing `evaluation_results.csv` without re-running retrieval/generation:

```bash
python generate_report.py
```

For a single all-in-one command (dependency check + full pipeline + report), useful in Colab or as a quick local smoke test:

```bash
python colab_run.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Covers Precision@K, Recall@K, MRR, Hit Rate, semantic similarity, faithfulness, retriever behavior, and end-to-end evaluator output (37 tests total).

## Optional Gemini Setup

1. Get a Google AI Studio API key.
2. Copy `.env.example` to `.env` and set:
   ```
   GOOGLE_API_KEY=your-key-here
   USE_LLM_GENERATION=true   # to use Gemini for answer generation
   USE_LLM_JUDGE=true        # to run the optional LLM-as-judge evaluator
   ```
3. Install the optional dependency: `pip install google-generativeai`.
4. Re-run `python run_evaluation.py`. If the API call fails for any reason, the pipeline logs it and continues with local/deterministic evaluation — LLM access is never required for the project to complete.

## Optional LangSmith Setup

LangSmith tracing is **not required** and is not wired into this project by default, to keep the dependency footprint small. If you want tracing on top of this project, the natural integration point is `src/generator.py::AnswerGenerator._try_llm_generate` and `src/llm_judge.py::judge_with_llm` — wrap those calls with LangSmith's `@traceable` decorator (`pip install langsmith`, set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`) if you want to inspect individual LLM calls.

---

## Example Output

```
============================================================
RAG EVALUATION REPORT (top_k=3)
===============================

Test Cases          : 15

Retrieval
Precision@K         : 0.62
Recall@K            : 0.87
MRR                 : 0.79
Hit Rate@K          : 0.87

Generation
Answer Accuracy     : 0.71
Faithfulness        : 0.88
Answer Relevance    : 0.69
Hallucination Rate  : 0.12

Performance
Mean Latency        : 0.0142s
Median Latency      : 0.0131s
P95 Latency         : 0.0210s

Cost
Estimated Tokens    : 3412
Estimated Cost      : $0.000000
============================================================
```

Numbers above are illustrative of the report *format*; your actual run will print real numbers computed from your environment (they vary depending on whether the real sentence-transformers model or the offline hashing fallback is used — see Limitations).

## Example Evaluation Table

| question_id | precision_at_k | recall_at_k | answer_accuracy | faithfulness_score | failure_category |
|---|---|---|---|---|---|
| q001 | 0.33 | 1.00 | 0.81 | 0.92 | correct_answer |
| q002 | 0.00 | 0.00 | 0.40 | 0.55 | retrieval_failure |
| q005 | 0.67 | 1.00 | 0.77 | 0.90 | correct_answer |
| q006 | 0.33 | 0.50 | 0.58 | 0.61 | generation_failure |

(A full, real version of this table is written to `results/evaluation_results.csv` and `results/error_analysis.csv` every run.)

---

## Limitations

- **Embedding-based metrics are proxies**, not ground truth — see "Limitations of embedding-based evaluation" above.
- **Hallucination detection here is a heuristic**, not a validated fact-checking system; it measures semantic overlap with retrieved context, not factual accuracy against the real world.
- **The offline hashing embedding fallback** (used only when sentence-transformers model weights can't be downloaded) is much weaker than real sentence embeddings and will noticeably lower all similarity-based scores; it exists purely so the pipeline is runnable in network-restricted environments, not as a quality baseline.
- **The knowledge base is intentionally small** (10 documents, 15 questions) to keep the project fast, free, and reviewable — it is a benchmark harness demonstration, not a large-scale IR benchmark.
- **Token counts are estimates** (`len(text.split())`), not exact provider tokenization, and are explicitly labeled `estimated_tokens` throughout the code and outputs.
- **Cost figures are configurable estimates** (`INPUT_COST_PER_1M_TOKENS` / `OUTPUT_COST_PER_1M_TOKENS`), not verified, current production API pricing.
- **The LLM-as-judge is optional and unverified** by this project — it is one more signal, not a source of truth, and can fail, be inconsistent, or be biased by prompt phrasing.

## Future Improvements

- Swap the word-window chunker for a sentence- or semantic-boundary-aware chunker.
- Add a proper vector index (e.g. FAISS) once the knowledge base grows beyond what fits comfortably in memory.
- Add inter-rater comparison between the deterministic faithfulness metric and the optional LLM judge to quantify how much they agree.
- Add confidence intervals / statistical significance testing to the experiment comparison rather than raw mean deltas.
- Support additional embedding models and make model choice part of the experiment grid, not just `top_k`.

## Engineering Lessons

- Deterministic and model-based metrics answer *different* questions, and conflating them (into a single "quality score") destroys the diagnostic value of an evaluation harness. Keeping retrieval metrics and generation metrics separate — and explicit about their respective methodology — was the most important design decision in this project.
- An evaluation pipeline is itself a piece of software that needs to fail gracefully: every optional dependency (the embedding model, the LLM generator, the LLM judge) has an explicit, tested fallback path so a missing API key or blocked network call degrades the output rather than crashing the run.
- Error analysis is more valuable than an aggregate score. A single "faithfulness: 0.88" number doesn't tell a team what to fix; a `failure_category` breakdown does.

## Author

Built as a portfolio project demonstrating RAG evaluation methodology for AI/LLM engineering roles. Contributions and forks welcome — see Installation above to get started locally or in Colab.
