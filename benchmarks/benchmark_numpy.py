import json
import time
from pathlib import Path

import numpy as np

from src.numpy_analysis import (
    calculate_vectorized_metrics,
    calculate_loop_metrics,
)


def benchmark(size, repeats=20, warmups=3):
    if size <= 0 or repeats <= 0 or warmups < 0:
        raise ValueError("Size and repeats must be positive; warmups cannot be negative.")

    rng = np.random.default_rng(42)
    values = rng.uniform(0, 100, size=size).astype(np.float64)

    vectorized_times = []
    loop_times = []

    # Warm-up phase: exclude initial execution overhead.
    for _ in range(warmups):
        calculate_vectorized_metrics(values)
        calculate_loop_metrics(values)

    vectorized = None
    loop = None

    # Alternate execution order to reduce systematic timing bias.
    for iteration in range(repeats):
        if iteration % 2 == 0:
            start = time.perf_counter()
            vectorized = calculate_vectorized_metrics(values)
            vectorized_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            loop = calculate_loop_metrics(values)
            loop_times.append(time.perf_counter() - start)
        else:
            start = time.perf_counter()
            loop = calculate_loop_metrics(values)
            loop_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            vectorized = calculate_vectorized_metrics(values)
            vectorized_times.append(time.perf_counter() - start)

    if vectorized["count"] != loop["count"]:
        raise AssertionError("Metric counts do not match.")

    for key in ("mean", "median", "minimum", "maximum", "total"):
        if not np.isclose(vectorized[key], loop[key]):
            raise AssertionError(f"Metric mismatch: {key}")

    vectorized_median = float(np.median(vectorized_times))
    loop_median = float(np.median(loop_times))

    return {
        "array_size": size,
        "warmups": warmups,
        "repeats": repeats,
        "vectorized_median_seconds": vectorized_median,
        "loop_median_seconds": loop_median,
        "speedup": (
            loop_median / vectorized_median
            if vectorized_median > 0 else None
        ),
        "results_match": True,
    }


if __name__ == "__main__":
    results = [
        benchmark(1_000),
        benchmark(10_000),
        benchmark(100_000),
    ]

    output = Path(__file__).resolve().parents[1] / "outputs"
    output.mkdir(exist_ok=True)

    report = output / "benchmark_results.json"
    report.write_text(json.dumps(results, indent=4), encoding="utf-8")

    print(json.dumps(results, indent=4))
    print(f"\nBenchmark saved to: {report}")
