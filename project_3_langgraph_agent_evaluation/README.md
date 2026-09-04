# LangGraph Agent Evaluation & Regression Framework

A small LangGraph agent evaluated like a software system: correctness, tool
selection, trajectory, reliability, latency, and regression between two
agent versions.

## Project Overview

Pipeline: `USER INPUT → ROUTER → TOOL → ANSWER → EVALUATOR`

The router sends each input to **knowledge search**, a **calculator**, or a
**direct** reply. The evaluator then scores the full run, not just the final
text.

## Project Structure
```
📁 langgraph-agent-evaluation/
├── 📁 data
│   └── 📄 dataset.json
├── 📁 notebook
│   └── 📄 langgraph_agent_evaluation.ipynb
|
├── 📁 screenshot  # screenshot while running project
├── 📁 results
│   ├── 📄 .gitkeep
│   ├── 📄 error_analysis.csv
│   ├── 📄 evaluation_results.csv
│   ├── 📄 regression_comparison.csv
│   └── 📄 regression_report.json
├── 📁 src
│   ├── 📁 __pycache__
│   │   ├── 📄 agent.cpython-313.pyc
│   │   ├── 📄 evaluator.cpython-313.pyc
│   │   ├── 📄 metrics.cpython-313.pyc
│   │   └── 📄 regression.cpython-313.pyc
│   ├── 📄 agent.py
│   ├── 📄 evaluator.py
│   ├── 📄 metrics.py
│   └── 📄 regression.py
├── 📁 tests
│   └── 📄 test_project.py
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 run_evaluation.py
```
## Architecture
run_evaluation.py # entry point
src/agent.py # LangGraph agent (router, tools, answer)
src/metrics.py # correctness, accuracy, reliability, latency
src/evaluator.py # runs the dataset through an agent version
src/regression.py # V1 vs V2 comparison + regression decision
data/dataset.json # 12 evaluation cases
results/ # CSV/JSON outputs (generated on run)
tests/test_project.py # pytest suite


## Why Agent Evaluation Matters

An agent can produce the correct final answer while using the wrong tool or
wrong reasoning path. Therefore, evaluating only the final answer is
insufficient — this framework also checks *which* tool was used and *what
path* the agent took to get there.

## Tools

- **Knowledge Search** — keyword matching over 7 small hard-coded documents
  about LLMs, RAG, embeddings, agents, hallucinations, and evaluation. No
  vector database.
- **Calculator** — supports `+ - * /` via a small regex-based parser. No
  `eval()`.

## Metrics

| Metric | Method |
|---|---|
| Answer correctness | Exact numeric match for calculator cases; `all-MiniLM-L6-v2` semantic similarity for text cases |
| Tool accuracy | `expected_tool == actual_tool` |
| Trajectory accuracy | `expected_path == actual_path` |
| Reliability | successful cases / total cases |
| Latency | mean, median, p95 via `time.perf_counter()` |

## Trajectory Evaluation

Each run's node path (`router → tool → answer`) is recorded, and `evaluate`
is appended by the harness. A case only counts as trajectory-correct if the
full path matches exactly — catching cases where the right answer was
reached by the wrong route.

## V1 vs V2

- **V1**: correct router (calculator pattern checked before knowledge keywords).
- **V2**: the check order is swapped, so inputs like `"Evaluate 12 * 8"` match
  the knowledge keyword `"evaluat"` first and get misrouted to knowledge
  instead of the calculator.

## Regression Detection

```python
MAX_DROP = 0.05
```

If V1 → V2 causes answer correctness, tool accuracy, trajectory accuracy, or
reliability to drop by more than `MAX_DROP`, the run prints
`REGRESSION DETECTED`; otherwise `NO REGRESSION`. Latency is reported but not
used to trigger regression.

## Error Analysis

`results/error_analysis.csv` lists every failed case with `case_id`, `input`,
`expected_tool`, `actual_tool`, `expected_path`, `actual_path`,
`answer_score`, `latency`, and `failure_reason`.

## Installation

```bash
pip install -r requirements.txt
python run_evaluation.py
```

## Google Colab Usage

Run `build_agent_evaluation_project.py` in one cell — it creates the full
project under `/content/langgraph-agent-evaluation` and a matching zip. Then:

```bash
%cd /content/langgraph-agent-evaluation
!pip install -q -r requirements.txt
!python run_evaluation.py
```

The first run downloads the `all-MiniLM-L6-v2` model (~80 MB) from
Hugging Face; after that it is cached and the framework runs fully offline.

## Testing

```bash
pytest tests/ -q
```

Covers the calculator, knowledge search, routing (V1 and the V2 bug),
answer correctness, trajectory evaluation, and regression detection.

## Limitations

- Keyword matching is not real NLU; wording outside the 7 documents' keywords
  won't be found.
- Only 12 dataset cases — enough to demonstrate the framework, not a full
  benchmark.
- The semantic similarity model needs one internet-connected run to download
  its weights.

## Future Improvements

- Larger, more diverse dataset.
- Pluggable tools and a real vector-based retriever.
- Track more agent versions over time instead of just V1 vs V2.
