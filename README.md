# Day 05 — NumPy Fundamentals

**Python Engineering Training | Halcer**

## Project Overview

This project demonstrates the use of NumPy for numerical data analysis and performance optimization using Python.

The application processes ticket resolution-time data from a CSV file, performs numerical operations using NumPy arrays, calculates statistical metrics, filters records, and compares vectorized operations against traditional Python loops.

The primary objective is to understand how NumPy improves numerical processing through efficient array operations, vectorization, broadcasting, and statistical calculations.

## Learning Objectives

- Understand NumPy arrays and their properties.
- Work with array shape, dimensions, size, and data types.
- Handle missing and invalid numerical values.
- Perform array filtering and conditional operations.
- Calculate statistical metrics using NumPy.
- Understand vectorization and broadcasting.
- Compare Python loops with NumPy vectorized operations.
- Write reusable and testable numerical analysis functions.
- Measure performance using repeatable benchmarks.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3 | Application development |
| NumPy | Numerical computation and array operations |
| Pytest | Automated testing |
| pytest-cov | Code coverage measurement |
| CSV | Input dataset format |
| JSON | Benchmark results storage |

## Project Structure

```text
Day 5/
│
├── benchmarks/
│   └── benchmark_numpy.py
│
├── data/
│   └── sample_tickets.csv
│
├── evidence/
│   ├── pipeline_execution.txt
│   ├── test_results.txt
│   ├── coverage_report.txt
│   └── benchmark_execution.txt
│
├── outputs/
│   └── benchmark_results.json
│
├── src/
│   ├── __init__.py
│   └── numpy_analysis.py
│
├── tests/
│   ├── __init__.py
│   └── test_numpy_analysis.py
│
├── DAILY_REPORT.md
├── README.md
└── requirements.txt
```

## Application Functionality

### 1. CSV Data Loading

- Reads ticket resolution-time records from a CSV file.
- Validates required columns.
- Handles missing and invalid numerical values.
- Converts numerical data into NumPy arrays.

### 2. NumPy Array Analysis

The application analyzes array properties, including:

- Shape
- Number of dimensions
- Data type
- Total number of elements

### 3. Missing Value Handling

The application uses `NaN` to represent missing numerical values and excludes invalid values from valid statistical calculations.

It also handles negative and non-finite numerical values through validation.

### 4. Statistical Analysis

The application calculates:

- Count
- Mean
- Median
- Minimum
- Maximum
- Total

These metrics are calculated for the overall valid resolution-time data and separately for each priority.

### 5. Filtering and Conditional Operations

The application uses NumPy filtering operations to select resolution-time values that satisfy specified conditions.

### 6. Vectorization

Vectorization allows numerical operations to be performed directly on NumPy arrays without explicitly iterating through every element using Python loops.

This reduces execution overhead and improves performance for numerical workloads.

### 7. Broadcasting

Broadcasting allows NumPy to perform operations between compatible arrays of different shapes without manually repeating data.

### 8. Performance Benchmarking

The benchmark compares traditional Python loop-based calculations against NumPy vectorized calculations.

The benchmark uses three array sizes:

- 1,000 elements
- 10,000 elements
- 100,000 elements

Each benchmark uses five repetitions and compares median execution times.

## Installation and Setup

### Prerequisites

- Python 3.10 or later
- pip
- Git

### Step 1: Clone the repository

```bash
git clone https://github.com/halcer-abhay-0005/Day_05_task_NumPy_Fundamentals.git
cd Day_05_task_NumPy_Fundamentals
```

### Step 2: Create a virtual environment

```bash
python3 -m venv .venv
```

### Step 3: Activate the environment

macOS / Linux:

```bash
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Run the NumPy analysis application:

```bash
python3 -m src.numpy_analysis
```

The application prints the analysis results in JSON format.

## Running Tests

Execute the complete test suite:

```bash
python3 -m pytest -v
```

Run tests with code coverage:

```bash
python3 -m pytest --cov=src --cov-report=term-missing
```

## Running Benchmarks

Execute the benchmark from the project root:

```bash
PYTHONPATH=. python3 benchmarks/benchmark_numpy.py
```

The benchmark results are saved to:

```text
outputs/benchmark_results.json
```

## Verification Results

The following results were obtained during local execution.

| Verification | Result |
|---|---:|
| Total input records | 20 |
| Valid numerical records | 18 |
| Automated tests | 28 passed |
| Code coverage | 96% |
| Benchmark array sizes | 3 |
| Benchmark repetitions | 5 |
| Benchmark result comparisons | All matched |

### Benchmark Results

| Array Size | Vectorized Median | Loop Median | Speed Advantage |
|---:|---:|---:|---:|
| 1,000 | 0.00002904 sec | 0.00046396 sec | ~15.98× |
| 10,000 | 0.00014421 sec | 0.00502438 sec | ~34.84× |
| 100,000 | 0.00096421 sec | 0.05078175 sec | ~52.67× |

**Observation:** NumPy vectorized operations were faster than traditional Python loops in all three benchmark scenarios. Both approaches produced matching results.

*Note: Benchmark timings are specific to the local execution environment and are not universal performance guarantees.*

## Testing and Quality Assurance

The automated test suite covers:

- CSV loading and validation.
- Array shape and data type.
- Missing values.
- Filtering operations.
- Vectorized statistical calculations.
- Loop and vectorized result comparison.
- Empty arrays.
- Negative and non-finite values.
- Broadcasting and incompatible shapes.
- Priority-wise calculations.
- Invalid files and CSV structures.
- Application execution.

**Latest verified test result:** 28 passed.

**Latest verified code coverage:** 96%.

Four statements remain uncovered in the current coverage report. The project has not been represented as having 100% coverage.

## Key Learnings

1. NumPy arrays are more efficient for numerical operations than manually processing individual values in many workloads.
2. Vectorization simplifies numerical code and reduces explicit Python loops.
3. Broadcasting makes compatible array operations easier.
4. Missing values require deliberate handling before statistical calculations.
5. Automated tests help verify numerical correctness and edge cases.
6. Benchmarking provides measurable evidence when comparing implementation approaches.

## Limitations

- The project currently uses a small sample CSV dataset.
- Benchmark timings depend on hardware and execution environment.
- The benchmark focuses on the implemented numerical calculations rather than every possible NumPy workload.
- The current test coverage is 96%.

## Author

**Abhay Gour**

Python And AI Engineering  | AI/ML Engineering Learning Track

**Repository:** [Day 05 — NumPy Fundamentals](https://github.com/halcer-abhay-0005/Day_05_task_NumPy_Fundamentals)