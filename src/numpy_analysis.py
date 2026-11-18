from pathlib import Path
import csv
import numpy as np


REQUIRED_COLUMNS = {
    "ticket_id",
    "priority",
    "category",
    "resolution_time_hours",
}


def load_ticket_data(file_path):
    """Load ticket records and validate the CSV structure."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV file is empty or has no header.")

        normalized_headers = [
            header.strip().lower() if header else ""
            for header in reader.fieldnames
        ]

        if any(not header for header in normalized_headers):
            raise ValueError("CSV contains an empty column name.")

        if len(normalized_headers) != len(set(normalized_headers)):
            raise ValueError(
                "CSV contains duplicate column names after normalization."
            )

        missing = REQUIRED_COLUMNS - set(normalized_headers)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        reader.fieldnames = normalized_headers

        records = []
        for row_number, record in enumerate(reader, start=2):
            if None in record:
                raise ValueError(
                    f"Malformed CSV row {row_number}: extra fields detected."
                )

            if any(value is None for value in record.values()):
                raise ValueError(
                    f"Malformed CSV row {row_number}: missing fields detected."
                )

            records.append(record)

    return records


def extract_resolution_times(records):
    """Convert resolution times into a one-dimensional float array."""
    values = []

    for record in records:
        raw_value = record["resolution_time_hours"]

        if raw_value is None or not raw_value.strip():
            values.append(np.nan)
            continue

        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            values.append(np.nan)
            continue

        values.append(value if np.isfinite(value) and value >= 0 else np.nan)

    return np.asarray(values, dtype=np.float64)


def inspect_array(values):
    """Return basic information about an array."""
    array = np.asarray(values)

    return {
        "shape": array.shape,
        "dimensions": array.ndim,
        "dtype": str(array.dtype),
        "size": array.size,
    }


def filter_resolutions(values, minimum=0, maximum=None):
    """Filter finite resolution values using boolean indexing."""
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")

    if not np.isfinite(minimum):
        raise ValueError("Minimum must be finite.")

    if maximum is not None:
        if not np.isfinite(maximum) or maximum < minimum:
            raise ValueError("Maximum must be finite and >= minimum.")

    mask = np.isfinite(array) & (array >= minimum)

    if maximum is not None:
        mask &= array <= maximum

    return array[mask]


def calculate_vectorized_metrics(values):
    """Calculate numerical metrics without Python loops."""
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")

    valid = array[np.isfinite(array)]

    if valid.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "total": 0.0,
        }

    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "minimum": float(np.min(valid)),
        "maximum": float(np.max(valid)),
        "total": float(np.sum(valid)),
    }


def calculate_loop_metrics(values):
    """Independent loop-based implementation for verification."""
    array = np.asarray(values, dtype=np.float64)

    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional array.")

    valid = [float(value) for value in array if np.isfinite(value)]

    if not valid:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
            "total": 0.0,
        }

    ordered = sorted(valid)
    count = len(ordered)
    middle = count // 2

    median = (
        ordered[middle]
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )

    return {
        "count": count,
        "mean": sum(valid) / count,
        "median": median,
        "minimum": min(valid),
        "maximum": max(valid),
        "total": sum(valid),
    }


def calculate_priority_metrics(records):
    """Calculate resolution metrics separately for each priority."""
    results = {}

    priorities = np.asarray(
        [record["priority"] for record in records],
        dtype=str,
    )
    resolutions = extract_resolution_times(records)

    for priority in np.unique(priorities):
        mask = priorities == priority
        results[str(priority)] = calculate_vectorized_metrics(
            resolutions[mask]
        )

    return results


def calculate_adjusted_values(values, adjustments):
    """Demonstrate NumPy broadcasting with explicit shape validation."""
    base = np.asarray(values, dtype=np.float64)
    adjustment = np.asarray(adjustments, dtype=np.float64)

    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(adjustment)):
        raise ValueError("Inputs must contain finite values.")

    try:
        return base + adjustment
    except ValueError as exc:
        raise ValueError(
            f"Incompatible shapes: {base.shape} and {adjustment.shape}"
        ) from exc


def run_analysis(file_path):
    records = load_ticket_data(file_path)
    resolutions = extract_resolution_times(records)

    return {
        "total_records": len(records),
        "array_info": inspect_array(resolutions),
        "valid_metrics": calculate_vectorized_metrics(resolutions),
        "priority_metrics": calculate_priority_metrics(records),
        "filtered_resolutions": filter_resolutions(
            resolutions, minimum=0, maximum=10
        ).tolist(),
    }


if __name__ == "__main__":
    import json

    dataset = Path(__file__).resolve().parents[1] / "data" / "sample_tickets.csv"

    results = run_analysis(dataset)

    print(json.dumps(results, indent=4))
