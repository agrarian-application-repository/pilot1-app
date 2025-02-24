import numpy as np


# ----------------------------------------------------
# Multi-Object Interaction Features
# ----------------------------------------------------

def compute_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between objects at each time step.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) where:
          - N is the number of objects,
          - 2 represents the x and y coordinates,
          - T is the number of time steps.

    Returns
    -------
    distances : np.ndarray
        Array of shape (T, N, N) where distances[t, i, j] is the Euclidean distance
        between object i and object j at time step t.
    """
    # Transpose positions to shape (T, N, 2) for easier per-time-step processing.
    pos_t = np.transpose(positions, (2, 0, 1))  # shape: (T, N, 2)
    # Use broadcasting to compute differences between every pair of objects.
    diff = pos_t[:, :, np.newaxis, :] - pos_t[:, np.newaxis, :, :]  # shape: (T, N, N, 2)
    distances = np.linalg.norm(diff, axis=3)  # shape: (T, N, N)
    return distances


def compute_pairwise_relative_velocity(velocity: np.ndarray) -> np.ndarray:
    """
    Compute pairwise relative velocity vectors between objects at each time step.

    The relative velocity for a pair (i, j) is defined as the difference between their velocity
    vectors: (velocity of i) - (velocity of j).

    Parameters
    ----------
    velocity : np.ndarray
        Array of shape (N, 2, T) representing velocity vectors for each object.
        (Typically obtained as the time derivative of positions.)

    Returns
    -------
    relative_velocity : np.ndarray
        Array of shape (T, N, N, 2) where for each time step t,
        relative_velocity[t, i, j, :] is the relative velocity vector between objects i and j.
    """
    # Transpose to shape (T, N, 2)
    vel_t = np.transpose(velocity, (2, 0, 1))  # shape: (T, N, 2)
    # Compute differences between every pair of velocity vectors.
    rel_vel = vel_t[:, :, np.newaxis, :] - vel_t[:, np.newaxis, :, :]  # shape: (T, N, N, 2)
    return rel_vel


def compute_pairwise_relative_speed(velocity: np.ndarray) -> np.ndarray:
    """
    Compute pairwise relative speed (the magnitude of the relative velocity) between objects.

    Parameters
    ----------
    velocity : np.ndarray
        Array of shape (N, 2, T) representing velocity vectors for each object.

    Returns
    -------
    relative_speed : np.ndarray
        Array of shape (T, N, N) where each element is the magnitude of the relative velocity
        between a pair of objects at each time step.
    """
    rel_vel = compute_pairwise_relative_velocity(velocity)  # shape: (T, N, N, 2)
    relative_speed = np.linalg.norm(rel_vel, axis=3)  # shape: (T, N, N)
    return relative_speed


def compute_local_density(positions: np.ndarray, radius: float) -> np.ndarray:
    """
    Compute the local density for each object by counting the number of neighboring objects
    within a specified radius.

    For each object at each time step, the density is defined as the count of other objects
    whose distance is less than the specified radius.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing the positions of N objects over T time steps.
    radius : float
        The threshold distance within which another object is considered a neighbor.

    Returns
    -------
    local_density : np.ndarray
        Array of shape (N, T) where each element represents the number of neighbors (excluding itself)
        for the corresponding object and time step.
    """
    # Compute pairwise distances: shape (T, N, N)
    distances = compute_pairwise_distances(positions)
    # For each time step and object, count neighbors with distance < radius.
    # Each object's self-distance is 0, so subtract 1.
    density_t = np.sum(distances < radius, axis=2) - 1  # shape: (T, N)
    # Transpose to shape (N, T) for consistency.
    local_density = density_t.T
    return local_density


def compute_average_nearest_neighbor_distance(positions: np.ndarray, k: int = 1) -> np.ndarray:
    """
    Compute the average distance to the k nearest neighbors for each object at each time step.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing the positions of N objects over T time steps.
    k : int, optional
        The number of nearest neighbors to consider (default is 1).

    Returns
    -------
    avg_nn_distance : np.ndarray
        Array of shape (N, T) containing the average distance to the k nearest neighbors for each object.
    """
    # Compute pairwise distances: shape (T, N, N)
    distances = compute_pairwise_distances(positions)
    T, N, _ = distances.shape
    # Prepare an array to hold the average nearest neighbor distances.
    avg_nn_distance = np.zeros((N, T))

    # For each time step, sort the distances for each object and compute the average of the k nearest (non-self) distances.
    for t in range(T):
        # distances[t] is shape (N, N)
        for i in range(N):
            sorted_dists = np.sort(distances[t, i, :])
            # The first entry is the self-distance (0); take the next k distances.
            k_effective = min(k, len(sorted_dists) - 1)
            if k_effective > 0:
                avg_nn_distance[i, t] = np.mean(sorted_dists[1:k_effective + 1])
            else:
                avg_nn_distance[i, t] = 0.0
    return avg_nn_distance


def compute_motion_correlation(time_series: np.ndarray) -> np.ndarray:
    """
    Compute the Pearson correlation coefficient of a motion metric (e.g., speed or acceleration)
    between each pair of objects.

    Parameters
    ----------
    time_series : np.ndarray
        Array of shape (N, T) representing a motion metric (such as speed) over time for N objects.

    Returns
    -------
    correlation_matrix : np.ndarray
        Array of shape (N, N) where correlation_matrix[i, j] is the Pearson correlation coefficient
        between the time series of object i and object j.
    """
    # np.corrcoef expects each variable as a row; if time_series is (N, T), this works directly.
    correlation_matrix = np.corrcoef(time_series)
    return correlation_matrix


# ----------------------------------------------------
# Example Usage
# ----------------------------------------------------
if __name__ == "__main__":
    # For demonstration, assume we have 4 objects, 2D positions, and 20 time steps.
    N = 4
    T = 20
    # Generate random positions in a normalized 2D space.
    positions = np.random.rand(N, 2, T)

    # 1. Compute Pairwise Distances
    distances = compute_pairwise_distances(positions)
    print("Pairwise Distances shape:", distances.shape)  # Expected: (T, N, N)

    # 2. Compute Pairwise Relative Velocity
    # First, compute velocity as the difference in positions (assuming dt=1).
    velocity = np.diff(positions, axis=2)  # shape: (N, 2, T-1)
    rel_velocity = compute_pairwise_relative_velocity(velocity)
    print("Pairwise Relative Velocity shape:", rel_velocity.shape)  # Expected: (T-1, N, N, 2)

    # 3. Compute Pairwise Relative Speed
    rel_speed = compute_pairwise_relative_speed(velocity)
    print("Pairwise Relative Speed shape:", rel_speed.shape)  # Expected: (T-1, N, N)

    # 4. Compute Local Density with a specified radius.
    radius = 0.2
    local_density = compute_local_density(positions, radius)
    print("Local Density shape:", local_density.shape)  # Expected: (N, T)

    # 5. Compute Average Nearest Neighbor Distance (using k=1).
    avg_nn_distance = compute_average_nearest_neighbor_distance(positions, k=1)
    print("Average Nearest Neighbor Distance shape:", avg_nn_distance.shape)  # Expected: (N, T)

    # 6. Compute Motion Correlation.
    # For instance, calculate speed (magnitude of velocity) from velocity.
    speed = np.linalg.norm(velocity, axis=1)  # shape: (N, T-1)
    motion_corr = compute_motion_correlation(speed)
    print("Motion Correlation shape:", motion_corr.shape)  # Expected: (N, N)
