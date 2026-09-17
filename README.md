# AI Evaluation Lab

**A portfolio of independent, reproducible evaluation frameworks for RAG pipelines, LLM production benchmarking, and LangGraph agents — built to demonstrate evaluation methodology, not just working demos.**

Most "AI project" portfolios stop at *"I built a chatbot that answers questions from my documents."* That proves plumbing works — it doesn't prove the system is any good, or that you know how to tell the difference. This lab exists to demonstrate the harder, more valuable skill: **designing metrics, running controlled experiments, and diagnosing *why* an AI system fails, not just *that* it did.**

Each project is self-contained, dependency-light, runs fully offline by default (no API key required), and ships with its own tests, plots, and detailed README.

---

## Projects

| # | Project | What it evaluates | Key techniques |
|---|---|---|---|
| 1 | [RAG Evaluation & Benchmarking System](./project_1_rag_evaluation) | A Retrieval-Augmented Generation pipeline — retrieval quality *and* generation quality, scored separately | Precision@K / Recall@K / MRR / Hit Rate, embedding-based faithfulness & hallucination proxy, `top_k` experiment comparison, per-example failure classification, optional Gemini LLM-as-judge |
| 2 | [LLM Performance & Cost Observatory](./project_2_llm_performance_observatory) | Multiple LLM configurations, compared on quality, latency, cost, and reliability | Quality via semantic similarity, exact-vs-estimated token accounting, configurable cost model, Pareto-efficiency analysis, weighted composite production score |
| 3 | [LangGraph Agent Evaluation & Regression Framework](./project_3_langgraph_agent_evaluation) | A routing agent (LangGraph) — final answer, tool choice, *and* execution trajectory | Answer correctness, tool accuracy, trajectory (path) accuracy, V1-vs-V2 regression detection with configurable failure threshold |

Every project follows the same shape: **fixed labeled dataset → pipeline run → deterministic + model-based metrics → CSV/JSON results + plots → error analysis → documented limitations.**

---

## Why This Repo Exists

Building an AI feature is easy to demo and hard to trust. These three projects each isolate one recurring evaluation problem in real AI engineering work:

- **RAG systems** fail in two independent places — retrieval and generation — and a single "quality" score hides which one broke. Project 1 measures them separately and shows *why* that distinction matters.
- **Choosing a production LLM** is a multi-objective tradeoff between quality, latency, cost, and reliability, not a leaderboard lookup. Project 2 makes that tradeoff explicit with Pareto analysis instead of picking a single "winner."
- **Agents** can land on the right final answer through the wrong path. Project 3 evaluates the trajectory, not just the output, and demonstrates regression detection between two agent versions.

Across all three, the same engineering principles are enforced deliberately:

- **Deterministic metrics where ground truth exists, model-based metrics where it doesn't** — and never the two conflated into one opaque "quality" number.
- **Graceful degradation.** Every optional dependency (an embedding model download, an LLM API call, an LLM-as-judge, tracing) has a tested fallback path. A missing API key or blocked network call never crashes a run — it's logged and the pipeline continues on deterministic ground truth.
- **Error analysis over aggregate scores.** Every run writes a per-example `failure_category` breakdown, because "faithfulness: 0.88" doesn't tell you what to fix — a categorized list of *which* examples failed and *how* does.
- **Honest limitations.** Every README explicitly documents what its metrics are proxies for, where the numbers come from, and what they don't prove.

---

## Tech Stack

- **Language:** Python 3.10+
- **Core:** NumPy, pandas, scikit-learn, matplotlib
- **Embeddings / semantic metrics:** `sentence-transformers` (`all-MiniLM-L6-v2`), with a deterministic offline fallback when the model can't be downloaded
- **Agents:** LangGraph (Project 3)
- **Optional model providers:** Google Gemini (`google-generativeai`) — used only for optional LLM-as-judge / LLM generation / real-mode benchmarking; never required
- **Optional observability:** LangSmith tracing (best-effort, never blocking)
- **Testing:** pytest (unit tests per project covering metrics, retrieval, benchmarking, regression detection)

No project requires an API key, GPU, or paid service to run its default configuration end to end.

---

## Repository Structure

```
ai-evaluation-lab/
├── project_1_rag_evaluation/                 # RAG retrieval + generation evaluation harness
│   ├── src/                                  # chunker, embeddings, retriever, generator, metrics, evaluator, llm_judge, reporting
│   ├── data/                                 # 10 knowledge-base docs + 15 labeled eval questions
│   ├── results/                              # generated CSV/JSON + plots
│   ├── tests/
│   └── README.md
│
├── project_2_llm_performance_observatory/    # Multi-model quality / latency / cost / reliability benchmark
│   ├── src/                                  # providers, benchmark, evaluator, metrics (Pareto + scoring), reporting
│   ├── data/                                 # 18 fixed benchmark questions across 5 categories
│   ├── results/
│   ├── tests/
│   └── README.md
│
├── project_3_langgraph_agent_evaluation/     # LangGraph router-agent evaluation + V1 vs V2 regression
│   ├── src/                                  # agent (router/tools/answer), metrics, evaluator, regression
│   ├── data/                                 # 12 labeled evaluation cases
│   ├── results/
│   ├── tests/
│   └── README.md
│
├── LICENSE                                   # MIT
└── README.md                                 # you are here
```

---

## Quick Start

Each project is independent — clone the repo and `cd` into whichever one you want:

```bash
git clone https://github.com/kunalkirtak/ai-evaluation-lab.git
cd ai-evaluation-lab/project_1_rag_evaluation      # or project_2_.../project_3_...

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python run_evaluation.py       # project_1 / project_3
# or
python run_benchmark.py        # project_2

python -m pytest tests/ -v
```

All three also include Google Colab instructions in their own READMEs for a zero-setup run.

See each project's README for its full architecture, metric definitions, methodology, example output, and documented limitations:

- **[Project 1 — RAG Evaluation & Benchmarking System →](./project_1_rag_evaluation/README.md)**
- **[Project 2 — LLM Performance & Cost Observatory →](./project_2_llm_performance_observatory/README.md)**
- **[Project 3 — LangGraph Agent Evaluation & Regression Framework →](./project_3_langgraph_agent_evaluation/README.md)**

---

## Engineering Lessons Across the Lab

- **Separate what you can measure exactly from what you can only estimate**, and label each accordingly (`token_source`, `EXACT_PROVIDER_TOKENS` vs `ESTIMATED_TOKENS`, deterministic vs embedding-based metrics). Conflating them destroys the diagnostic value of an evaluation harness.
- **An evaluation pipeline is software and needs to fail gracefully.** Every optional dependency across all three projects — embedding models, LLM judges, real providers, tracing — has an explicit, tested fallback so the run degrades instead of crashing.
- **A single score is a summary, not a diagnosis.** Trajectory checks, failure categories, and Pareto fronts exist because "0.88" or "best model: X" hides the tradeoffs a team actually needs to see.

## Roadmap

- Shared vector index (FAISS) once any project's knowledge base grows beyond in-memory scale.
- Additional LLM provider adapters (OpenAI, Anthropic, local Ollama) across Projects 1 and 2.
- Statistical significance testing on experiment comparisons, rather than raw mean deltas.
- Track more than two agent versions over time in Project 3, instead of a single V1-vs-V2 comparison.

## License

MIT — see [LICENSE](./LICENSE).

## Author

Built by [Kunal Kirtak](https://github.com/kunalkirtak) as a portfolio demonstrating AI/LLM evaluation engineering: metric design, controlled experimentation, cost and reliability modeling, and clear technical communication. Contributions and forks welcome — see Quick Start above.
