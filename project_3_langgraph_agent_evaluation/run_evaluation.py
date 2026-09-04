"""Entry point: run Agent V1 and Agent V2 over the dataset, evaluate, and check for regressions."""
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.evaluator import evaluate_agent, load_dataset, save_results  # noqa: E402
from src.regression import compare, save_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = ROOT / "data" / "dataset.json"
RESULTS_DIR = ROOT / "results"


def fmt(value: float) -> str:
    return f"{value:.3f}"


def print_block(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    print(f"Answer Correctness: {fmt(metrics['answer_correctness'])}")
    print(f"Tool Accuracy:       {fmt(metrics['tool_accuracy'])}")
    print(f"Trajectory Accuracy: {fmt(metrics['trajectory_accuracy'])}")
    print(f"Reliability:         {fmt(metrics['reliability'])}")
    print(f"Mean Latency:        {metrics['mean_latency'] * 1000:.2f} ms")


def main() -> None:
    dataset = load_dataset(str(DATA_PATH))
    logger.info("Loaded %d evaluation cases", len(dataset))

    logger.info("Evaluating Agent V1 (correct router)...")
    v1 = evaluate_agent("v1", buggy=False, dataset=dataset)

    logger.info("Evaluating Agent V2 (buggy router)...")
    v2 = evaluate_agent("v2", buggy=True, dataset=dataset)

    save_results(str(RESULTS_DIR), v1, v2)

    report = compare(v1["metrics"], v2["metrics"])
    save_report(str(RESULTS_DIR), report)

    print("=== AGENT EVALUATION ===")
    print_block("V1", v1["metrics"])
    print_block("V2", v2["metrics"])
    print("\n=== REGRESSION ===")
    print(f"Status: {report['status']}")
    print(f"Reason: {report['reason']}")
    print(f"\nResults written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
