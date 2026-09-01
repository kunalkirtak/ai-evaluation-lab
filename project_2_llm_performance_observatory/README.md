# LLM Performance & Cost Observatory

A compact, reproducible benchmarking and observability framework for comparing LLM configurations across **quality, latency, token usage, cost, throughput, and reliability** — and turning those measurements into a defensible production recommendation.

> **Simulated by default.** Out of the box this project runs entirely in a local, deterministic simulation mode. No API key, no network access (after installing dependencies), and no cost is required to see the full pipeline — dataset → generation → evaluation → cost accounting → comparison report → plots — run end to end.

---

## 1. Executive Summary

Teams shipping LLM features constantly face one question: **"Which model / configuration should we actually use for this workload?"** The instinctive answer — "use the most powerful model" — ignores latency budgets, per-request cost at scale, and reliability under load.

This project builds a small but complete **benchmarking engine** that runs a fixed evaluation dataset against multiple model configurations, measures them along every dimension that matters in production, and produces a **reproducible comparison report** with a measurement-based recommendation — including a Pareto-efficiency analysis so the tradeoffs are visible, not hidden behind a single ranking number.

## 2. Why LLM Observability Matters

Unlike traditional software, LLM systems have **non-deterministic, open-ended outputs**, **variable latency**, and **usage-based cost** that scales with every request. Without systematic measurement, teams end up choosing models based on vibes, marketing benchmarks, or whichever demo looked good — not on data that reflects their actual workload, prompts, and quality bar.

## 3. Problem Statement

Given a fixed set of representative questions, benchmark several candidate LLM configurations and answer:

- Which configuration is fastest?
- Which is cheapest per request / per successful answer?
- Which produces the highest-quality answers?
- Which is most reliable?
- Which configurations are **Pareto-efficient** — i.e., not strictly dominated by another option on quality, latency, and cost simultaneously?
- Given configurable priorities, which configuration is the **best balanced** choice?

## 4. Key Engineering Questions

1. How do you measure "quality" for open-ended text without a human in the loop for every run?
2. How do you separate *exact* provider-reported token usage from *estimated* token counts?
3. How do you keep cost assumptions honest and clearly labeled as configurable, not universal fact?
4. How do you keep one failed request from crashing an entire benchmark run?
5. How do you turn four competing metrics (quality, latency, cost, reliability) into one actionable recommendation without hiding the tradeoffs?

## 5. Architecture

```
Dataset (18 fixed questions)
        │
        ▼
Benchmark Runner  ───────────────►  LLMProvider (interface)
        │                                 │
        │                         ┌───────┴────────┐
        │                         │                 │
        │                 LocalMockProvider   GeminiProvider (optional)
        │                  (default, no key)   (USE_REAL_MODE=true)
        ▼
  Generation Result (answer, tokens, latency, success/error)
        │
        ▼
  Quality Evaluation (semantic similarity + relevance, sentence-transformers)
        │
        ▼
  Cost Calculation (configurable $/1M tokens)
        │
        ▼
  Reliability + Throughput Aggregation
        │
        ▼
  Comparison Report: CSV + JSON + PNG plots + terminal summary
        │
        ▼
  Pareto Analysis + Composite Production Score + Recommendation
```

The benchmark runner depends **only** on the `LLMProvider` abstract interface — never directly on Gemini or any specific SDK — so real-model integration is fully optional and swappable.

## 6. Benchmark Pipeline

`src/benchmark.py::run_benchmark()` executes, **sequentially** (by design, for simplicity and reproducibility):

1. Load the fixed dataset (`data/benchmark_dataset.json`).
2. For each configured provider, for each question: call `generate()`.
3. Evaluate the generated answer against the reference answer (semantic similarity + relevance).
4. Calculate cost from token counts and the configured pricing.
5. Classify the request into a failure category (`successful`, `api_error`, `timeout`, `low_quality`, `validation_error`).
6. Aggregate all per-request records into per-model comparison rows.
7. Run Pareto-efficiency analysis and compute a composite production score.
8. Write CSV/JSON/plots and print a terminal report.

## 7. Metrics

| Category | Metrics |
|---|---|
| Quality | correctness, relevance, semantic similarity |
| Performance | request / mean / median / P95 latency |
| Usage | input, output, total tokens |
| Economics | cost per request, total cost, cost per successful answer, cost per quality point |
| Reliability | success rate, error rate, failure count by category |
| Throughput | requests per second |

## 8. Quality Evaluation

Default evaluation is **fully local** and requires no external LLM call:

- **Semantic similarity** — generated answer vs. reference answer, via `sentence-transformers` (`all-MiniLM-L6-v2` by default). Cosine similarity is mapped from `[-1, 1]` to `[0, 1]`.
- **Answer relevance** — question vs. generated answer, same embedding model.
- **Correctness threshold** — a configurable `QUALITY_THRESHOLD` (default `0.70`) above which an answer is classified as correct for this benchmark. This is a **project-defined experiment setting**, not an industry standard.
- If `sentence-transformers` cannot load (e.g. no internet), the evaluator **automatically falls back** to a lightweight bag-of-words Jaccard similarity so the pipeline never breaks — it just uses a cruder quality signal, and this is logged clearly.

An **optional LLM-as-judge** (`ENABLE_LLM_JUDGE=true`, requires `GOOGLE_API_KEY`) can additionally score correctness/relevance/completeness via a structured JSON prompt. It is never required, and any failure is caught and logged without interrupting the run. Note that LLM-as-judge introduces its own evaluator bias and should be interpreted with that in mind.

## 9. Latency Measurement

Every request is timed with `time.perf_counter()`. Aggregates computed: mean, median, P95, min, max. In local mode, latency is **simulated** per configuration profile (fast/balanced/high-quality) purely to exercise the analysis pipeline — it is **not** a measurement of any real commercial model's speed.

## 10. Token Accounting

- **Real mode (Gemini):** uses provider-reported `usage_metadata` when available → labeled `EXACT_PROVIDER_TOKENS`.
- **Local mode:** uses a simple estimator, `max(1, len(text.split()))` → labeled `ESTIMATED_TOKENS`.

Every record stores which source was used (`token_source` column) so the two are never silently conflated.

## 11. Cost Model

Cost is computed from a **configurable** pricing assumption (`$ per 1,000,000 tokens`), set per model in `run_benchmark.py` or via `.env`:

```
input_cost  = (input_tokens  / 1,000,000) * input_price_per_1m
output_cost = (output_tokens / 1,000,000) * output_price_per_1m
total_cost  = input_cost + output_cost
```

**These are example, configurable benchmark assumptions — not a guarantee of any real provider's current pricing.** Always check the provider's official pricing page before using this for real budgeting.

## 12. Reliability Metrics

```
success_rate = successful_requests / total_requests
error_rate   = failed_requests / total_requests
```

Failures are further broken down into `api_error`, `timeout`, `low_quality`, and `validation_error` counts.

## 13. Throughput

`requests_per_second = total_requests / benchmark_duration_seconds`. Execution is sequential in this project (documented explicitly) — a concurrent mode would inflate throughput but adds complexity that isn't needed to demonstrate the measurement methodology.

## 14. Multi-Model Experiments

Three simulated local configurations ship by default:

| Configuration | Profile |
|---|---|
| `local-fast-lowcost` | Low latency, lower quality, cheapest pricing assumption |
| `local-balanced` | Middle ground on all axes |
| `local-highquality` | Higher latency and cost, highest simulated quality |

**These are simulated benchmark profiles for demonstrating the framework — they are not measurements of any real commercial model.** If `USE_REAL_MODE=true` and `GOOGLE_API_KEY` is set, a real Gemini configuration is benchmarked alongside a local baseline instead.

## 15. Pareto Analysis

A configuration is **Pareto-efficient** if no other configuration is at least as good on quality, latency, *and* cost while being strictly better on at least one of them. In other words, you cannot improve one metric without giving up something on another — these are the configurations worth considering; anything else is dominated and can be discarded. See `src/metrics.py::pareto_efficient`.

## 16. Composite Production Score

A configurable, **project-defined** (not industry-standard) weighted score combining quality, reliability, normalized speed, and normalized cost-efficiency:

```
production_score = quality_weight      * quality
                  + reliability_weight  * reliability
                  + latency_weight      * normalized_speed
                  + cost_weight         * normalized_cost_efficiency
```

Default weights (override via `.env`): `QUALITY_WEIGHT=0.50`, `RELIABILITY_WEIGHT=0.20`, `LATENCY_WEIGHT=0.15`, `COST_WEIGHT=0.15`. Latency and cost are min-max normalized and inverted (lower is better → higher normalized score) before combining, so higher `production_score` always means "better."

## 17. Error Analysis

`results/error_analysis.csv` records every non-successful request (failed calls **and** low-quality answers below the correctness threshold), with the question, generated vs. reference answer, quality score, latency, error message, and failure category — useful for spotting systematic weaknesses rather than just an aggregate error rate.

## 18. Observability

Optional [LangSmith](https://www.langchain.com/langsmith) tracing (`ENABLE_LANGSMITH=true` + `LANGSMITH_TRACING=true`) wraps each request in a best-effort trace context. If the SDK is missing, misconfigured, or the environment variables aren't set, the benchmark proceeds completely normally — LangSmith is never required.

## 19. Project Structure

```
llm-performance-observatory/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── run_benchmark.py
├── build_llm_observatory_project.py   # regenerates this entire repo
├── src/
│   ├── __init__.py
│   ├── config.py          # env-driven configuration, pricing, weights
│   ├── models.py           # shared dataclasses
│   ├── dataset.py          # loads the fixed benchmark dataset
│   ├── providers.py        # LLMProvider interface + Local/Gemini adapters
│   ├── evaluator.py        # semantic similarity, relevance, optional judge
│   ├── metrics.py           # latency stats, cost, reliability, Pareto, score
│   ├── benchmark.py        # orchestrates the full pipeline
│   ├── reporting.py        # CSV/JSON/plots/terminal report
│   └── observability.py    # optional LangSmith integration
├── data/
│   └── benchmark_dataset.json   # 18 fixed questions across 5 categories
├── results/                # generated on run (gitignored)
│   └── plots/
└── tests/
    ├── test_metrics.py
    ├── test_cost.py
    ├── test_benchmark.py
    └── test_evaluator.py
```

## 20. Installation

```bash
git clone <your-repo-url>
cd llm-performance-observatory
pip install -r requirements.txt
```

No API key is required for the default local mode.

## 21. Google Colab Instructions

1. Open a new Google Colab notebook.
2. Upload `llm-performance-observatory.zip` (or clone the repo) and extract it:
   ```python
   !unzip -q llm-performance-observatory.zip -d /content
   %cd /content/llm-performance-observatory
   ```
3. Install requirements:
   ```python
   !pip install -q -r requirements.txt
   ```
4. Run the benchmark:
   ```python
   !python run_benchmark.py
   ```
5. Run the tests:
   ```python
   !python -m pytest tests/ -v
   ```
6. Inspect results: `results/*.csv`, `results/summary.json`, and the plots in `results/plots/` (open them with `from IPython.display import Image; Image("results/plots/quality_vs_cost.png")`).
7. Download the repository / zip results for your portfolio.
8. Upload the folder to GitHub as a new repository.

## 22. VS Code Instructions

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_benchmark.py
python -m pytest tests/ -v
```

## 23. Running the Benchmark

```bash
python run_benchmark.py
```

Produces:

```
results/benchmark_results.csv     # every raw request
results/model_comparison.csv      # aggregated per-model comparison
results/error_analysis.csv        # failed / low-quality requests
results/summary.json              # overall summary + recommendation
results/run_config.json           # reproducibility metadata
results/plots/*.png               # 5 comparison charts
```

## 24. Running Tests

```bash
python -m pytest tests/ -v
```

Covers token estimation, cost calculation, percentile/latency stats, success/error rate, quality scoring, benchmark aggregation, Pareto analysis, and the provider interface — no real API calls required.

## 25. Optional: Real Gemini Configuration

1. Copy `.env.example` to `.env`.
2. Set `GOOGLE_API_KEY=<your key>` and `USE_REAL_MODE=true`.
3. Run `python run_benchmark.py` — a real `gemini-1.5-flash` configuration will be benchmarked alongside a local baseline. Any API failure is caught, logged, and recorded as a failed request; it will not crash the run.

## 26. Optional: LangSmith Configuration

1. Set `ENABLE_LANGSMITH=true` and `LANGSMITH_TRACING=true` (plus `LANGSMITH_API_KEY` per LangSmith's own setup docs).
2. Install `langsmith` (already in `requirements.txt`).
3. Run the benchmark normally — traces are best-effort and never block execution.

## 27. Example Output

```
============================================================
LLM PERFORMANCE OBSERVATORY
===========================

Benchmark Mode: LOCAL_SIMULATION

Configurations: 3
Test Cases: 18

QUALITY
Best Quality: local-highquality (0.661)

PERFORMANCE
Lowest Mean Latency: local-fast-lowcost (0.0251s)

ECONOMICS
Lowest Cost: local-fast-lowcost ($0.000011/request)

RELIABILITY
Best Success Rate: local-balanced (100.0%)

PARETO-EFFICIENT CONFIGURATIONS
local-balanced, local-fast-lowcost, local-highquality

RECOMMENDATION
Best balanced (composite production score): local-fast-lowcost
Note: 'best' depends on your workload's priorities. Review
results/model_comparison.csv and results/plots/quality_vs_cost.png
before deciding.

============================================================
```

*(Your exact numbers will vary slightly depending on whether `sentence-transformers` can download its model in your environment; a deterministic fallback similarity is used automatically if not.)*

## 28. Example Comparison Table

| model | success_rate | mean_quality | mean_latency | cost_per_request | production_score |
|---|---|---|---|---|---|
| local-fast-lowcost | 1.00 | 0.51 | 0.025s | $0.000011 | 0.75 |
| local-balanced | 1.00 | 0.59 | 0.050s | $0.000095 | 0.62 |
| local-highquality | 0.89 | 0.66 | 0.050s | $0.000518 | 0.51 |

## 29. Limitations

- Benchmark results depend entirely on the workload (dataset, prompts, domain) — they do not generalize to arbitrary tasks.
- Latency in real mode varies with network conditions, provider load, and infrastructure; a single run is not a definitive speed measurement.
- Token estimation in local mode is a word-count heuristic, not equivalent to any provider's real tokenizer/billing units.
- Cost figures depend entirely on the configured pricing assumptions, which are examples and must be updated to match real, current provider pricing.
- Local simulation mode does **not** represent the performance of any real commercial LLM — it exists purely to demonstrate the benchmarking methodology without requiring API access.
- Embedding-based semantic similarity is a proxy for quality, not a perfect judge of correctness.
- An optional LLM-as-judge can introduce its own bias and inconsistency.
- A higher-quality model is not automatically the right production choice — quality, latency, cost, and reliability must be weighed together against the workload's actual requirements.

## 30. Future Improvements

- Optional concurrent execution mode for higher-throughput benchmarking.
- Additional provider adapters (OpenAI, Anthropic, local Ollama models).
- Statistical significance testing across repeated runs.
- A richer LLM-as-judge rubric with per-category breakdowns.
- Historical run comparison / regression detection across benchmark runs over time.

## 31. Engineering Lessons

Building this project reinforced that **production LLM decisions are multi-objective optimization problems**, not leaderboard lookups. A clean provider abstraction made it trivial to swap between a zero-cost local simulation and a real API without touching the benchmarking engine — and defensive error handling (catch, log, continue) throughout the pipeline meant a single failed request never invalidated an entire experiment. Making every pricing assumption, quality threshold, and scoring weight explicitly configurable (rather than hard-coded "facts") kept the framework honest about what it does and does not measure.

## 32. Author

Built as a portfolio project demonstrating production AI engineering practices: benchmarking methodology, evaluation design, cost modeling, reliability engineering, and clear technical communication.
