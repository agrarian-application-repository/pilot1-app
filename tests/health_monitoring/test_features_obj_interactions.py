import numpy as np

from src.health_monitoring.timeseries.objects_interactions import (
    compute_pairwise_distances,
    compute_local_density,
    compute_average_nearest_neighbor_distance,
)


def test_compute_pairwise_distances_basic():
    # 3 objects, 2D, 2 time steps
    # shape (3, 2, 2) => (N=3, 2D, T=2)
    positions = np.array([
        [[0, 0],    # [x1, x2]
         [1, 1]],   # [y1, y2]

        [[1, 0],    # [x1, x2]
         [1, 0]],   # [y1, y2]

        [[0, 2],    # [x1, x2]
         [3, 5]],   # [y1, y2]
    ])

    # shape (T=2, N=3, N=3)
    expected = np.array([
        [[0.0,              1.0,            2.0],
         [1.0,              0.0,            np.sqrt(5.0)],
         [2.0,              np.sqrt(5.0),   0.0]],  # t=0

        [[0.0,              1.0,                np.sqrt(20.0)],
         [1.0,              0.0,                np.sqrt(29.0)],
         [np.sqrt(20.0),    np.sqrt(29.0),      0.0]],  # t=1
    ])

    distances = compute_pairwise_distances(positions)
    np.testing.assert_allclose(distances, expected, rtol=1e-6)


def test_compute_local_density_basic():
    # 3 objects, 2 time step
    distances = np.array([
        [
            [0.0, 0.5, 1.5],
            [0.5, 0.0, 2.0],
            [1.5, 2.0, 0.0],
        ],
        [
            [0.0, 2.5, 3.0],
            [2.5, 0.0, 4.0],
            [3.0, 4.0, 0.0],
        ],

    ])  # shape (T=2, N=3, N=3)

    radius = 1.0
    expected_density = np.array([
        [1, 0],  # object 0
        [1, 0],  # object 1
        [0, 0],  # object 2
    ])  # shape (N=3, T=2)

    density = compute_local_density(distances, radius)
    np.testing.assert_array_equal(density, expected_density), density

    radius = 2.5
    expected_density = np.array([
        [2, 1],  # object 0
        [2, 1],  # object 1
        [2, 0],  # object 2
    ])  # shape (N=3, T=2)

    density = compute_local_density(distances, radius)
    np.testing.assert_array_equal(density, expected_density), density


def test_local_density_zero_radius():
    distances = np.array([
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 4.5],
            [1.0, 4.5, 0.0],
        ],
        [
            [0.0, 2.0, 3.0],
            [2.0, 0.0, 4.5],
            [3.0, 4.5, 0.0],
        ],
    ])  # shape (T=2, N=3, N=3)

    result = compute_local_density(distances, radius=0.0)
    expected = np.array([
        [0, 0],
        [0, 0],
        [0, 0],
    ])

    np.testing.assert_array_equal(result, expected)


def test_average_nearest_neighbor_distance_basic():
    # 3 objects, 2 time step
    distances = np.array([
        [
            [0.0, 0.5, 1.5],
            [0.5, 0.0, 2.0],
            [1.5, 2.0, 0.0],
        ],
        [
            [0.0, 2.5, 3.0],
            [2.5, 0.0, 4.0],
            [3.0, 4.0, 0.0],
        ],

    ])  # shape (T=2, N=3, N=3)

    k = 1
    expected_k1 = np.array([
        [0.5, 2.5],  # object 0
        [0.5, 2.5],  # object 1
        [1.5, 3.0],  # object 2
    ])  # shape (N=3, T=1)

    result_k1 = compute_average_nearest_neighbor_distance(distances, k=k)
    np.testing.assert_allclose(result_k1, expected_k1)

    k = 2
    expected_k2 = np.array([
        [1.0, 2.75],
        [1.25, 3.25],
        [1.75, 3.5],
    ])

    result_k2 = compute_average_nearest_neighbor_distance(distances, k=k)
    np.testing.assert_allclose(result_k2, expected_k2)


def test_average_nearest_neighbor_distance_k_larger_than_num_elements():
    distances = np.array([
        [
            [0.0, 1.0, 4.0],
            [1.0, 0.0, 3.5],
            [4.0, 3.5, 0.0],
        ],
        [
            [0.0, 3.0, 1.0],
            [3.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
    ])  # shape (T=2, N=3, N=3)

    k = 4
    result = compute_average_nearest_neighbor_distance(distances, k=k)
    expected = np.array([
        [2.5, 2.0],
        [2.25, 2.0],
        [3.75, 1.0],
    ])

    np.testing.assert_allclose(result, expected)



