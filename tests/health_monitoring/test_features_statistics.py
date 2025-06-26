import numpy as np
from scipy.stats import circmean, circstd

from src.health_monitoring.timeseries.temporal_statistics import (
    compute_temporal_linear_statistics,
    compute_temporal_circular_statistics,
    compute_average_timeseries,
    compute_median_timeseries,
    compute_circular_average_timeseries
)


def test_compute_temporal_linear_statistics_array():
    data = np.array([   # (2,4)
        [1, 2, 3, 4],
        [4, 4, 4, 4]
    ])

    stats = compute_temporal_linear_statistics(data, return_array=True)
    assert stats.shape == (2, 6)
    expected = np.array([
        [np.mean(data[0]), np.median(data[0]), np.std(data[0]), np.min(data[0]), np.max(data[0]), 1.5],
        [np.mean(data[1]), np.median(data[1]), np.std(data[1]), np.min(data[1]), np.max(data[1]), 0.0],
    ])
    np.testing.assert_almost_equal(stats, expected)


def test_compute_temporal_circular_statistics_array():
    angles = np.array([
        [0, np.pi, -np.pi],
        [np.pi/2, -np.pi/2, 0]
    ])
    result = compute_temporal_circular_statistics(angles, return_array=True)
    assert result.shape == (2, 2)
    expected = np.array([
        [np.pi],
        [0],
    ])
    np.testing.assert_almost_equal(result[:, 0:1], expected)


def test_compute_average_timeseries_2d():
    values = np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])
    result = compute_average_timeseries(values)
    expected = np.array([2.5, 3.5, 4.5])
    np.testing.assert_array_equal(result, expected)


def test_compute_average_timeseries_3d():
    values = np.array([
        [[1, 2, 3], [4, 5, 6]],
        [[4, 5, 6], [7, 8, 9]],
    ])  # shape: (2, 2, 3)
    result = compute_average_timeseries(values)
    expected = np.array([
        [2.5, 3.5, 4.5], [5.5, 6.5, 7.5]
    ])
    np.testing.assert_array_equal(result, expected)


def test_compute_median_timeseries_2d():
    values = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
    ])  # shape: (2, 2, 1)
    result = compute_median_timeseries(values)
    expected = np.array([3, 4])
    np.testing.assert_array_equal(result, expected)


def test_compute_median_timeseries_3d():
    values = np.array([
        [[1, 2], [5, 6]],
        [[3, 4], [7, 8]],
        [[2, 1], [6, 4]],
    ])  # shape: (3, 2, 2)
    result = compute_median_timeseries(values)
    expected = np.array([
        [2, 2], [6, 6]
    ])
    np.testing.assert_array_equal(result, expected)


def test_compute_circular_average_timeseries():
    values = np.array([
        [np.pi, -np.pi, 0],
        [0, 0, 0]
    ])  # shape: (2, 3)
    result = compute_circular_average_timeseries(values)
    expected = circmean(values, axis=0, high=np.pi, low=-np.pi)
    np.testing.assert_allclose(result, expected)
