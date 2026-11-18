import json
import time
from pathlib import Path

import numpy as np

from src.numpy_analysis import (
    calculate_vectorized_metrics,
    calculate_loop_metrics,
)


def benchmark(size, repeats=5):
    rng = np.random.default_rng(42)
    values = rng.uniform(0, 100, size=size).astype(np.float64)

    vectorized_times = []
    loop_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        vectorized = calculate_vectorized_metrics(values)
        vectorized_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        loop = calculate_loop_metrics(values)
        loop_times.append(time.perf_counter() - start)

    assert vectorized["count"] == loop["count"]

    for key in ("mean", "median", "minimum", "maximum", "total"):
        assert np.isclose(vectorized[key], loop[key])

    return {
        "array_size": size,
        "repeats": repeats,
        "vectorized_median_seconds": float(np.median(vectorized_times)),
        "loop_median_seconds": float(np.median(loop_times)),
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
