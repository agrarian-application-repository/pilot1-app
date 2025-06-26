import numpy as np
from src.health_monitoring.timeseries.path_based import compute_cumulative_distance, compute_net_displacement, compute_path_efficiency


def test_compute_cumulative_distance_basic():
    ds = np.array([
        [1, 2, 3],  # obj1 displacement magnitudes
        [0, 0, 1],  # obj2 displacement magnitudes
        [2, 1, 4],  # obj3 displacement magnitudes
        [6, 1, 2],  # obj4 displacement magnitudes
    ])
    expected = np.array([
        [0, 1, 3, 6],
        [0, 0, 0, 1],
        [0, 2, 3, 7],
        [0, 6, 7, 9],
    ])
    result = compute_cumulative_distance(ds)
    np.testing.assert_array_equal(result, expected)


def test_compute_net_displacement_straight_line():
    # Two objects, 3 time steps
    pos_xy = np.array([
        [[0, 1, 2], [0, 0, 0]],  # Moves in x direction [x1, x2, x3], [y1, y2, y3]
        [[1, 2, 4], [1, 1, 1]]   # Moves in x direction
    ])
    expected = np.array([
        [0, 1, 2],
        [0, 1, 3]
    ])
    result = compute_net_displacement(pos_xy)
    np.testing.assert_allclose(result, expected)


def test_compute_net_displacement_diagonal():
    pos_xy = np.array([
        [[0, 1, 2], [0, 1, 2]],  # Moves diagonally
        [[1, 3, 7], [1, 3, 7]],  # Moves diagonally
    ])
    expected = np.array([
        [np.sqrt((0-0) ** 2 + (0-0) ** 2), np.sqrt((1-0) ** 2 + (1-0) ** 2), np.sqrt((2-0) ** 2 + (2-0) ** 2)],
        [np.sqrt((1-1) ** 2 + (1-1) ** 2), np.sqrt((3-1) ** 2 + (3-1) ** 2), np.sqrt((7-1) ** 2 + (7-1) ** 2)]
    ])
    result = compute_net_displacement(pos_xy)
    np.testing.assert_allclose(result, expected)


def test_compute_path_efficiency_direct_path():
    cum_dist = np.array([[0.0, 1.0, 2.0, 3.0]])
    net_disp = np.array([[0.0, 1.0, 2.0, 3.0]])
    expected = np.array([[0.0, 1.0, 1.0, 1.0]])
    result = compute_path_efficiency(cum_dist, net_disp)
    np.testing.assert_allclose(result, expected)


def test_compute_path_efficiency_indirect_path():
    cum_dist = np.array([[0.0, 2.0, 4.0, 6.0]])
    net_disp = np.array([[0.0, 1.0, 2.0, 3.0]])
    expected = np.array([[0.0, 0.5, 0.5, 0.5]])
    result = compute_path_efficiency(cum_dist, net_disp)
    np.testing.assert_allclose(result, expected)


def test_compute_path_efficiency_with_zeros():
    cum_dist = np.array([[0.0, 0.0, 0.0]])
    net_disp = np.array([[0.0, 1.0, 2.0]])
    expected = np.array([[0.0, 0.0, 0.0]])
    result = compute_path_efficiency(cum_dist, net_disp)
    np.testing.assert_array_equal(result, expected)
