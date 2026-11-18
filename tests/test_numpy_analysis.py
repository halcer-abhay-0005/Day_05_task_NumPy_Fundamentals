import numpy as np
import pytest

from src.numpy_analysis import (
    load_ticket_data,
    extract_resolution_times,
    inspect_array,
    filter_resolutions,
    calculate_vectorized_metrics,
    calculate_loop_metrics,
    calculate_priority_metrics,
    calculate_adjusted_values,
    run_analysis,
)


DATA = "data/sample_tickets.csv"


def test_load_ticket_data():
    records = load_ticket_data(DATA)
    assert len(records) == 20


def test_array_shape_and_dtype():
    records = load_ticket_data(DATA)
    values = extract_resolution_times(records)

    info = inspect_array(values)

    assert info["shape"] == (20,)
    assert info["dimensions"] == 1
    assert info["dtype"] == "float64"


def test_missing_values_are_nan():
    records = load_ticket_data(DATA)
    values = extract_resolution_times(records)

    assert np.isnan(values).sum() == 2


def test_filter_resolutions():
    values = np.array([2.0, 5.0, 12.0, np.nan, -1.0])

    result = filter_resolutions(values, minimum=0, maximum=5)

    np.testing.assert_array_equal(result, [2.0, 5.0])


def test_vectorized_metrics():
    values = np.array([2.0, 4.0, 6.0, np.nan])

    result = calculate_vectorized_metrics(values)

    assert result["count"] == 3
    assert result["mean"] == 4.0
    assert result["median"] == 4.0
    assert result["total"] == 12.0


def test_loop_and_vectorized_results_match():
    values = np.array([2.5, 4.5, 8.0, np.nan, 10.0])

    vectorized = calculate_vectorized_metrics(values)
    loop = calculate_loop_metrics(values)

    assert vectorized["count"] == loop["count"]

    for key in ("mean", "median", "minimum", "maximum", "total"):
        assert vectorized[key] == pytest.approx(loop[key])


def test_empty_array():
    result = calculate_vectorized_metrics(np.array([]))

    assert result["count"] == 0
    assert result["mean"] is None


def test_all_missing_values():
    result = calculate_vectorized_metrics(
        np.array([np.nan, np.nan])
    )

    assert result["count"] == 0
    assert result["mean"] is None


def test_negative_values_excluded():
    values = extract_resolution_times([
        {
            "resolution_time_hours": "-5",
        }
    ])

    assert np.isnan(values[0])


def test_incompatible_dimensions():
    with pytest.raises(ValueError):
        filter_resolutions(np.ones((2, 2)))


def test_invalid_filter_range():
    with pytest.raises(ValueError):
        filter_resolutions(np.array([1.0, 2.0]), minimum=5, maximum=2)


def test_broadcasting():
    result = calculate_adjusted_values(
        np.array([[1], [2], [3]]),
        np.array([10, 20]),
    )

    expected = np.array([
        [11, 21],
        [12, 22],
        [13, 23],
    ])

    np.testing.assert_array_equal(result, expected)


def test_incompatible_broadcasting_shapes():
    with pytest.raises(ValueError, match="Incompatible shapes"):
        calculate_adjusted_values(
            np.ones((3, 2)),
            np.ones((4,)),
        )


def test_priority_metrics():
    records = load_ticket_data(DATA)
    results = calculate_priority_metrics(records)

    assert set(results) == {"Critical", "High", "Medium", "Low"}
    assert results["Critical"]["count"] == 4


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_ticket_data("data/nonexistent.csv")


def test_missing_required_column(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text(
        "ticket_id,priority,category\nTKT001,High,Technical\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        load_ticket_data(path)


def test_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_ticket_data(path)


def test_duplicate_headers(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "ticket_id,priority,category,resolution_time_hours,"
        "resolution_time_hours\n"
        "TKT001,High,Technical,2,3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_ticket_data(path)


def test_loop_metrics_empty():
    result = calculate_loop_metrics([])

    assert result["count"] == 0
    assert result["total"] == 0.0


def test_non_finite_values():
    values = np.array([1.0, np.inf, -np.inf, np.nan, 3.0])

    result = calculate_vectorized_metrics(values)

    assert result["count"] == 2
    assert result["mean"] == 2.0


def test_two_dimensional_metrics_rejected():
    with pytest.raises(ValueError):
        calculate_vectorized_metrics(np.ones((2, 2)))


def test_non_finite_adjustment_rejected():
    with pytest.raises(ValueError):
        calculate_adjusted_values(
            np.array([1, 2]),
            np.array([np.inf, 3]),
        )


def test_loop_metrics_reject_two_dimensional_array():
    with pytest.raises(ValueError, match="one-dimensional"):
        calculate_loop_metrics(np.ones((2, 2)))


def test_invalid_numeric_resolution_becomes_nan():
    records = [
        {"resolution_time_hours": "unknown"},
        {"resolution_time_hours": "5.5"},
    ]

    result = extract_resolution_times(records)

    assert np.isnan(result[0])
    assert result[1] == 5.5


@pytest.mark.parametrize("minimum", [np.inf, -np.inf, np.nan])
def test_non_finite_minimum_rejected(minimum):
    with pytest.raises(ValueError, match="Minimum must be finite"):
        filter_resolutions(np.array([1.0, 2.0]), minimum=minimum)


def test_run_analysis():
    results = run_analysis(DATA)

    assert results["total_records"] == 20
    assert results["array_info"]["shape"] == (20,)
    assert results["valid_metrics"]["count"] == 18
    assert set(results["priority_metrics"]) == {
        "Critical", "High", "Medium", "Low"
    }
    assert len(results["filtered_resolutions"]) == 13
