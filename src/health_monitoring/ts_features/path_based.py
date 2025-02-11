import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ==============================
# Path-Based Metrics Functions
# ==============================

def compute_cumulative_distance(positions: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative distance traveled for each object over time.

    The function calculates the Euclidean distance between consecutive positions,
    then computes the cumulative sum along the time axis.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing positions of N objects over T time steps.

    Returns
    -------
    cumulative_distance : np.ndarray
        Array of shape (N, T) where each element represents the cumulative distance
        traveled up to that time step. The first time step is 0.
    """
    # Compute displacement between consecutive positions: shape (N, 2, T-1)
    displacement = np.diff(positions, axis=2)
    # Compute Euclidean distance for each displacement: shape (N, T-1)
    dist = np.linalg.norm(displacement, axis=1)
    # Prepend a column of zeros (starting at zero distance)
    zeros = np.zeros((positions.shape[0], 1))
    cumulative_distance = np.concatenate([zeros, np.cumsum(dist, axis=1)], axis=1)
    return cumulative_distance


def compute_net_displacement(positions: np.ndarray) -> np.ndarray:
    """
    Compute the net displacement for each object over time.

    Net displacement is defined as the straight-line distance from the initial position
    to the current position at each time step.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing positions.

    Returns
    -------
    net_displacement : np.ndarray
        Array of shape (N, T) representing the net displacement from the starting point.
    """
    # Difference between current positions and the initial position (broadcasting along time)
    displacement_from_start = positions - positions[:, :, 0:1]
    net_displacement = np.linalg.norm(displacement_from_start, axis=1)
    return net_displacement


def compute_path_efficiency(positions: np.ndarray) -> np.ndarray:
    """
    Compute the path efficiency for each object over time.

    Path efficiency is the ratio of the net displacement (straight-line distance from
    the start) to the cumulative distance traveled. Values near 1 indicate a direct path,
    while lower values suggest a more circuitous trajectory.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing positions.

    Returns
    -------
    efficiency : np.ndarray
        Array of shape (N, T) representing the path efficiency at each time step.
        When the cumulative distance is 0, efficiency is set to 0.
    """
    cum_distance = compute_cumulative_distance(positions)  # shape: (N, T)
    net_disp = compute_net_displacement(positions)  # shape: (N, T)
    # Avoid division by zero by specifying a small threshold
    efficiency = np.divide(net_disp, cum_distance, out=np.zeros_like(net_disp), where=cum_distance > 1e-6)
    return efficiency


def compute_turning_angle(positions: np.ndarray) -> np.ndarray:
    """
    Compute the turning angle between consecutive displacement vectors for each object.

    For each triplet of consecutive positions (p_t, p_{t+1}, p_{t+2}), the turning angle
    is computed as the angle between the displacement vectors (p_{t+1} - p_t) and
    (p_{t+2} - p_{t+1}) using the arctan2 formulation.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing positions.

    Returns
    -------
    turning_angle : np.ndarray
        Array of shape (N, T-2) representing the turning angles (in radians) for each
        applicable time step.
    """
    # Calculate displacement between consecutive positions: shape (N, 2, T-1)
    displacement = np.diff(positions, axis=2)
    # v1: displacement from time t to t+1, shape (N, 2, T-2)
    v1 = displacement[:, :, :-1]
    # v2: displacement from time t+1 to t+2, shape (N, 2, T-2)
    v2 = displacement[:, :, 1:]
    # Dot product along the coordinate axis
    dot_product = np.sum(v1 * v2, axis=1)
    # In 2D, the cross product (a scalar) is computed as:
    cross_product = v1[:, 0, :] * v2[:, 1, :] - v1[:, 1, :] * v2[:, 0, :]
    turning_angle = np.arctan2(cross_product, dot_product)
    return turning_angle


def compute_curvature(positions: np.ndarray) -> np.ndarray:
    """
    Compute the curvature of the trajectory for each object.

    Curvature is estimated as the turning angle divided by the arc length over the segment.
    For each triplet of positions, the arc length is approximated as the average of the
    lengths of two consecutive displacement vectors.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing positions.

    Returns
    -------
    curvature : np.ndarray
        Array of shape (N, T-2) representing the curvature (in radians per unit distance)
        for each applicable time step.
    """
    # Compute displacement and its magnitude
    displacement = np.diff(positions, axis=2)  # shape: (N, 2, T-1)
    disp_magnitude = np.linalg.norm(displacement, axis=1)  # shape: (N, T-1)

    # Turning angle computed from three consecutive positions: shape (N, T-2)
    turning_angle = compute_turning_angle(positions)

    # Approximate arc length as the average of consecutive displacement magnitudes
    arc_length = (disp_magnitude[:, :-1] + disp_magnitude[:, 1:]) / 2.0  # shape: (N, T-2)

    # Compute curvature safely (avoiding division by zero)
    curvature = np.divide(turning_angle, arc_length, out=np.zeros_like(turning_angle), where=arc_length > 1e-6)
    return curvature


# ==============================
# Example Usage
# ==============================
if __name__ == "__main__":
    # Assume we have 3 objects, positions in 2D space (normalized between 0 and 1), and 10 time steps.
    N = 3
    T = 10
    positions = np.random.rand(N, 2, T)

    # ----- Path-Based Metrics -----
    cumulative_distance = compute_cumulative_distance(positions)
    net_displacement = compute_net_displacement(positions)
    path_efficiency = compute_path_efficiency(positions)
    turning_angle = compute_turning_angle(positions)
    curvature = compute_curvature(positions)

    print("Cumulative Distance shape:", cumulative_distance.shape)  # (N, T)
    print("Net Displacement shape:", net_displacement.shape)  # (N, T)
    print("Path Efficiency shape:", path_efficiency.shape)  # (N, T)
    print("Turning Angle shape:", turning_angle.shape)  # (N, T-2)
    print("Curvature shape:", curvature.shape)  # (N, T-2)
