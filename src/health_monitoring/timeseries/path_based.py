import numpy as np
from src.health_monitoring.timeseries.istantaneous_kinematic import compute_magnitude


# ==============================
# Path-Based Metrics Functions
# ==============================

def compute_cumulative_distance(ds: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative distance traveled for each object over time,
    using the precomputed displacement magnitudes.

    Parameters
    ----------
    ds : np.ndarray
        Array of shape (N, T-1) representing the displacement magnitudes
        (i.e., Euclidean distances between consecutive positions).

    Returns
    -------
    cumulative_distance : np.ndarray
        Array of shape (N, T), where each element represents the cumulative distance
        traveled up to that time step. The first time step is 0.
    """
    # Prepend a column of zeros (starting at zero distance)
    zeros = np.zeros((ds.shape[0], 1))  # Shape: (N, 1)

    # Compute cumulative distance
    cumulative_distance = np.concatenate([zeros, np.cumsum(ds, axis=1)], axis=1)  # Shape: (N, T)

    return cumulative_distance


def compute_net_displacement(pos_xy: np.ndarray) -> np.ndarray:
    """
    Compute the net displacement for each object over time.

    Net displacement is defined as the straight-line distance from the initial position
    to the current position at each time step.

    Parameters
    ----------
    pos_xy : np.ndarray
        Array of shape (N, 2, T) representing positions.

    Returns
    -------
    net_displacement : np.ndarray
        Array of shape (N, T) representing the net displacement from the starting point.
    """
    # Difference between current positions and the initial position (broadcasting along time)
    displacement_from_start = pos_xy - pos_xy[:, :, 0:1]
    net_displacement = compute_magnitude(displacement_from_start)
    return net_displacement


def compute_path_efficiency(cumulative_distance: np.ndarray, net_displacement: np.ndarray) -> np.ndarray:
    """
    Compute the path efficiency for each object over time.

    Path efficiency is the ratio of the net displacement (straight-line distance from
    the start) to the cumulative distance traveled. Values near 1 indicate a direct path,
    while lower values suggest a more circuitous trajectory.

    Parameters
    ----------
    cumulative_distance : np.ndarray
        Array of shape (N, T) representing the precomputed cumulative distance traveled.

    net_displacement : np.ndarray
        Array of shape (N, T) representing the precomputed net displacement (straight-line distance from the start).

    Returns
    -------
    efficiency : np.ndarray
        Array of shape (N, T) representing the path efficiency at each time step.
        When the cumulative distance is 0, efficiency is set to 0.
    """
    # Avoid division by zero using np.divide with a threshold
    efficiency = np.divide(
        net_displacement,
        cumulative_distance,
        out=np.zeros_like(net_displacement),
        where=cumulative_distance > 1e-6
    )

    return efficiency


def compute_curvature(ds: np.ndarray, turning_angle: np.ndarray) -> np.ndarray:
    """
    Compute the curvature of the trajectory for each object.

    Curvature is estimated as the turning angle divided by the arc length over the segment.
    The arc length is approximated as the average of the lengths of two consecutive
    displacement vectors.

    Parameters
    ----------
    ds : np.ndarray
        Array of shape (N, T-1) representing the displacement magnitudes
        (Euclidean distances between consecutive positions).

    turning_angle : np.ndarray
        Array of shape (N, T-2) representing the turning angles (in radians).

    Returns
    -------
    curvature : np.ndarray
        Array of shape (N, T-2) representing the curvature (in radians per unit distance)
        for each applicable time step.
    """
    # Approximate arc length as the average of two consecutive displacement magnitudes
    arc_length = (ds[:, :-1] + ds[:, 1:]) / 2.0  # shape: (N, T-2)

    # Compute curvature safely (avoiding division by zero)
    curvature = np.divide(turning_angle, arc_length, out=np.zeros_like(turning_angle), where=arc_length > 1e-6)
    return curvature


# ==============================
# Example Usage
# ==============================
if __name__ == "__main__":
    from src.health_monitoring.timeseries.istantaneous_kinematic import compute_deltaS_xy, compute_deltaS_magnitude
    from src.health_monitoring.timeseries.directional import compute_turning_angle

    # Create a dummy dataset:
    N = 3  # Number of objects
    T = 5  # Number of time steps

    # Generate random positions (assuming values between 0 and 1 for a normalized space)
    positions = np.random.rand(N, 2, T)
    print("positions:", positions)

    ds_xy = compute_deltaS_xy(positions)
    ds = compute_deltaS_magnitude(ds_xy)
    turning_angle = compute_turning_angle(ds_xy)

    print("\ndisplacemt magnitude:", ds)

    # Compute cumulative distance
    cum_distance = compute_cumulative_distance(ds)
    print("\ncumulative_distance:", cum_distance)
    assert cum_distance.shape == (N, T), f"Expected 'cumulative_distance' shape {(N, T)}, got {cum_distance.shape}"

    # Compute net displacement
    net_disp = compute_net_displacement(positions)
    print("\nnet_displacement:", net_disp)
    assert net_disp.shape == (N, T), f"Expected 'net_displacement' shape {(N, T)}, got {net_disp.shape}"

    # Compute path efficiency
    efficiency = compute_path_efficiency(cum_distance, net_disp)
    print("\npath_efficiency:", efficiency)
    assert efficiency.shape == (N, T), f"Expected 'efficiency' shape {(N, T)}, got {efficiency.shape}"

    # Compute curvature
    curvature = compute_curvature(ds, turning_angle)
    print("\ncurvature:", curvature)
    assert curvature.shape == (N, T-2), f"Expected 'curvature' shape {(N, T-2)}, got {curvature.shape}"

