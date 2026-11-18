# Day 05 Daily Report — NumPy Fundamentals

- **Developer:** Abhay Gour
- **Assignment:** Day 05 — NumPy Fundamentals
- **Date:** 2026-11-18
- **Repository URL:** [https://github.com/halcer-abhay-0005/Day_05_task_NumPy_Fundamentals](https://github.com/halcer-abhay-0005/Day_05_task_NumPy_Fundamentals)
- **Latest commit hash:** Pending — documentation and initial Git commit not yet completed
- **Pull Request:** N/A — no pull request created

---

## 1. Objective

The objective of Day 05 was to understand NumPy fundamentals and apply numerical computing concepts to a practical ticket-data analysis problem.

The task focused on understanding array operations, numerical data handling, statistical calculations, vectorization, broadcasting, and performance benchmarking.

The implementation was designed to demonstrate the advantages of NumPy over traditional Python loops for numerical workloads.

## 2. Topics Learned

### A. NumPy Fundamentals

- Introduction to NumPy and its applications.
- Creating and working with NumPy arrays.
- Understanding array shape, dimensions, size, and data types.
- Understanding the difference between Python lists and NumPy arrays.

### B. Array Operations

- Array indexing and filtering.
- Conditional selection.
- Numerical calculations.
- Handling missing values using NaN.
- Working with empty and invalid arrays.

### C. Statistical Operations

- Mean calculation.
- Median calculation.
- Minimum and maximum values.
- Sum and count operations.
- Group-based numerical analysis.

### D. Vectorization

- Understanding vectorized operations.
- Comparing vectorized calculations with traditional Python loops.
- Understanding why vectorization improves numerical processing performance.

### E. Broadcasting

- Understanding NumPy broadcasting rules.
- Performing operations between compatible arrays.
- Handling incompatible array dimensions.

### F. Performance Benchmarking

- Measuring execution time.
- Comparing loop-based and vectorized implementations.
- Running benchmarks on different array sizes.
- Comparing results to ensure numerical correctness.

## 3. Implementation Completed

The following functionality was implemented:

1. Loaded ticket resolution-time data from a CSV file.
2. Converted numerical data into NumPy arrays.
3. Analyzed array shape, dimensions, size, and data type.
4. Handled missing and invalid numerical values.
5. Filtered resolution-time values using NumPy conditions.
6. Calculated statistical metrics for valid records.
7. Generated priority-wise numerical statistics.
8. Implemented vectorized calculations.
9. Compared loop-based and vectorized calculation results.
10. Implemented broadcasting examples.
11. Created a benchmark script to compare execution performance.
12. Added automated tests for normal cases and edge cases.

## 4. Application Execution Results

The application was executed successfully.

### Input Data Summary

| Metric | Result |
|---|---:|
| Total records | 20 |
| Valid numerical records | 18 |
| Mean resolution time | 7.97 hours |
| Median resolution time | 7.25 hours |
| Minimum resolution time | 1.5 hours |
| Maximum resolution time | 18 hours |
| Total resolution time | 143.5 hours |

### Priority-wise Analysis

| Priority | Count | Mean Resolution Time |
|---|---:|---:|
| Critical | 4 | 2.25 hours |
| High | 5 | 5.90 hours |
| Medium | 5 | 9.20 hours |
| Low | 4 | 14.75 hours |

## 5. Testing and Validation

The complete automated test suite was executed.

**Command:**

```bash
python3 -m pytest -v
```

**Result: 28 passed**

The test cases covered:

- CSV data loading.
- Array shape and data types.
- Missing values.
- Filtering operations.
- Vectorized calculations.
- Loop and vectorized result comparison.
- Empty arrays.
- Negative values.
- Non-finite numerical values.
- Broadcasting.
- Incompatible array shapes.
- Priority-wise calculations.
- Invalid CSV structures.
- Application execution.

### Code Coverage

**Command:**

```bash
python3 -m pytest --cov=src --cov-report=term-missing
```

| Metric | Result |
|---|---:|
| Total statements | 95 |
| Covered statements | 91 |
| Uncovered statements | 4 |
| Code coverage | 96% |

The current implementation achieved 96% code coverage.

## 6. Performance Benchmarking

A separate benchmark script was created to compare NumPy vectorized calculations against traditional Python loop-based calculations.

The benchmark was executed using five repetitions for each array size.

### Benchmark Results

| Array Size | Vectorized Time | Loop Time | Speed Advantage |
|---:|---:|---:|---:|
| 1,000 | 0.00002904 sec | 0.00046396 sec | 15.98× |
| 10,000 | 0.00014421 sec | 0.00502438 sec | 34.84× |
| 100,000 | 0.00096421 sec | 0.05078175 sec | 52.67× |

### Observations

- Vectorized calculations were faster in all three benchmark scenarios.
- Both approaches produced matching numerical results.
- The performance advantage increased with larger array sizes in these experiments.
- NumPy is particularly useful when performing repeated numerical operations on larger datasets.

## 7. Issues Encountered and Resolutions

### Issue: Benchmark Import Error

**Error:**

```text
ModuleNotFoundError: No module named 'src'
```

**Root Cause:**

When the benchmark script was executed directly, Python did not automatically include the project root directory in its module search path.

**Resolution:**

The benchmark was executed with the project root added to `PYTHONPATH`.

```bash
PYTHONPATH=. python3 benchmarks/benchmark_numpy.py
```

**Final Result:** Benchmark execution completed successfully, and the output was saved to `outputs/benchmark_results.json`.

## 8. Evidence Generated

The following evidence files were generated and verified:

```text
evidence/
├── pipeline_execution.txt
├── test_results.txt
├── coverage_report.txt
└── benchmark_execution.txt

outputs/
└── benchmark_results.json
```

These files contain the application output, test execution results, coverage information, and benchmark measurements.

## 9. Key Learnings

By completing this task, I gained practical understanding of:

1. How NumPy arrays store and process numerical data.
2. How array shape and dimensions affect numerical operations.
3. How to handle missing and invalid numerical values.
4. How to perform statistical calculations efficiently.
5. How vectorization reduces the need for explicit Python loops.
6. How broadcasting works with compatible array dimensions.
7. How to benchmark different implementations.
8. How automated tests help validate numerical calculations and edge cases.

## 10. Areas for Improvement

- Improve the remaining uncovered code paths and target higher test coverage.
- Experiment with larger datasets to understand performance differences.
- Explore advanced NumPy indexing and array manipulation.
- Practise more complex broadcasting operations.
- Explore memory-efficient numerical processing.

## 11. Next Learning Plan

- Continue strengthening Python engineering fundamentals.
- Practise advanced NumPy operations.
- Explore Pandas for structured data manipulation.
- Understand how NumPy and Pandas work together in data-processing pipelines.
- Apply numerical computing concepts to practical data engineering and machine learning problems.

## 12. Final Status

| Activity | Status |
|---|---|
| NumPy application implementation | Completed |
| Application execution | Passed |
| Automated tests | 28/28 passed |
| Code coverage | 96% |
| Benchmark execution | Passed |
| Numerical result comparison | Passed |
| Evidence generation | Completed |
| README documentation | Prepared |
| Daily report documentation | Prepared |
| Git initialization and push | Pending |

**Overall Result:** Day 05 implementation, testing, and benchmarking completed successfully. Documentation is prepared for final verification before Git submission.

---

**Developer:** Abhay Gour  
**Training:** Python And AI Engineering — Halcer