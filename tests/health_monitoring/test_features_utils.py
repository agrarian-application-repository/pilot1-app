import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from src.health_monitoring.timeseries.istantaneous_kinematic import (
    compute_delta_over_timestep,
    compute_magnitude,
    compute_direction,
    compute_time_derivative,
    encode_direction
)


def test_compute_delta_over_timestep():
    input_array = np.array([
        [[1, 2, 4], [0, 0, 0]],  # [x1,x2,x3], [y1, y2, y3]
        [[4, 5, 8], [0, 1, 2]]
    ])  # N=2, C=2, T=3

    expected_output = np.array([
        [[1, 2], [0, 0]],
        [[1, 3], [1, 1]]
    ])  # N=2, C=2, T=2

    output = compute_delta_over_timestep(input_array)
    assert_array_equal(output, expected_output)


def test_compute_magnitude():
    vectors = np.array([
        [[3, 0], [4, 1]],  # Euclidean norm: sqrt(3^2 + 4^2) = 5
        [[0, 5], [0, 12]]  # Euclidean norm: sqrt(5^2 + 12^2) = 13
    ])
    expected_output = np.array([
        [5, 1],
        [0, 13]
    ])
    output = compute_magnitude(vectors)
    assert_array_almost_equal(output, expected_output)


def test_compute_direction():
    vectors = np.array([
        [[1, 0], [0, 1]],  # angle(1,0) = 0, angle(0,1) = pi/2
        [[-1, 0], [0, -1]]  # angle(-1,0) = pi, angle(0,-1) = -pi/2
    ])
    expected_output = np.array([
        [0.0, np.pi / 2],
        [np.pi, -np.pi / 2]
    ])
    output = compute_direction(vectors)
    assert_array_almost_equal(output, expected_output)


def test_compute_time_derivative():
    delta_array = np.array([
        [[2, 4], [6, 8]],
        [[4, 12], [6, 18]]
    ])
    dt = 2.0
    expected_output = np.array([
        [[1.0, 2.0], [3.0, 4.0]],
        [[2.0, 6.0], [3.0, 9.0]],

    ])
    output = compute_time_derivative(delta_array, dt)
    assert_array_equal(output, expected_output)


def test_encode_direction():
    direction_angles = np.array([
        [0.0, np.pi / 2],  # sin(0)=0, cos(0)=1; sin(pi/2)=1, cos(pi/2)=0
        [np.pi, -np.pi / 2]  # sin(pi)=0, cos(pi)=-1; sin(-pi/2)=-1, cos(-pi/2)=0
    ])
    expected_output = np.array([
        [[0.0, 1.0], [1.0, 0.0]],   # [sinx1, sinx2], [cosx1, cosx2]
        [[0.0, -1.0], [-1.0, 0.0]]
    ])
    output = encode_direction(direction_angles)
    assert_array_almost_equal(output, expected_output, decimal=6)


# PYTHONPATH=. pytest tests/health_monitoring/test_features_utils.py (from inside AGRARIAN/)
