import numpy as np


def compute_heading(vel_xy: np.ndarray) -> np.ndarray:
    """
    Compute the heading (orientation) angle from velocity vectors.

    The heading is calculated as the angle between the velocity vector and
    the positive x-axis, using the arctan2 function to account for the correct
    quadrant. The angle is returned in radians.

    Parameters
    ----------
    vel_xy : np.ndarray
        Array of shape (N, 2, T) representing velocity vectors for N objects over T time steps.
        The first channel (index 0) is the x-component and the second (index 1) is the y-component.

    Returns
    -------
    heading : np.ndarray
        Array of shape (N, T) containing heading angles (in radians) for each object at each time step.
    """
    heading = np.arctan2(vel_xy[:, 1, :], vel_xy[:, 0, :])
    return heading


def compute_angular_velocity(heading: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute the angular velocity (rate of change of heading) for each object.

    Angular velocity is computed as the difference between consecutive unwrapped
    heading angles divided by the time step dt. Unwrapping helps to avoid issues
    with discontinuities (e.g., jumping from π to -π).

    Parameters
    ----------
    heading : np.ndarray
        Array of shape (N, T) representing heading angles (in radians) for each object over T time steps.
    dt : float, optional
        Time interval between consecutive heading measurements (default is 1.0).

    Returns
    -------
    angular_velocity : np.ndarray
        Array of shape (N, T-1) representing the angular velocity for each object over time.
    """
    # Unwrap the angles along the time axis to remove discontinuities
    unwrapped_heading = np.unwrap(heading, axis=1)
    # Compute the difference between consecutive headings and normalize by dt
    angular_velocity = np.diff(unwrapped_heading, axis=1) / dt
    return angular_velocity


def compute_turning_angle(ds_xy: np.ndarray) -> np.ndarray:
    """
    Compute the turning angle between consecutive displacement vectors for each object.

    The turning angle is a measure of how much the direction changes from one segment
    of the trajectory to the next. It is computed using the arctan2 of the cross product
    and dot product of two consecutive displacement (or velocity) vectors.

    For each object, if v1 is the displacement vector from time t to t+1 and
    v2 is from t+1 to t+2, then the turning angle at time t is:

        turning_angle = arctan2(cross(v1, v2), dot(v1, v2))

    where:
        cross(v1, v2) = v1_x * v2_y - v1_y * v2_x
        dot(v1, v2)   = v1_x * v2_x + v1_y * v2_y

    Parameters
    ----------
    ds_xy : np.ndarray
        Array of shape (N, 2, T-1) representing the displacement vectors between
        consecutive time steps for each object.

    Returns
    -------
    turning_angle : np.ndarray
        Array of shape (N, T-2) representing the turning angle (in radians) between consecutive displacement vectors.
        For each object, the turning angle is computed for each triple of consecutive positions.
    """
    # For consecutive displacement vectors, define:
    # v1 = displacement from time t to t+1 (shape: (N, 2, T-2))
    # v2 = displacement from time t+1 to t+2 (shape: (N, 2, T-2))
    # i.e., at the same index, v2 contains the displacement at the successive timestep w.r.t. v1
    v1 = ds_xy[:, :, :-1]
    v2 = ds_xy[:, :, 1:]

    # Compute the dot product and cross product (2D cross product returns a scalar)
    dot_product = v1[:, 0, :] * v2[:, 0, :] + v1[:, 1, :] * v2[:, 1, :]
    cross_product = v1[:, 0, :] * v2[:, 1, :] - v1[:, 1, :] * v2[:, 0, :]

    # Compute turning angle using arctan2
    turning_angle = np.arctan2(cross_product, dot_product)
    return turning_angle


if __name__ == "__main__":
    from src.health_monitoring.timeseries.istantaneous_kinematic import compute_deltaS_xy, compute_vel_xy
    # Create a dummy dataset:
    # Let's assume we have 3 objects, positions in 2D space, and 10 time steps.
    N = 3
    T = 10
    dt = 2.0
    # For reproducibility, generate random positions between 0 and 1.
    positions = np.random.rand(N, 2, T)
    print("positions:", positions)

    # To test directional metrics, first compute the velocity vectors from positions.
    # Displacement vectors (deltaS_xy) have shape (N, 2, T-1)
    ds_xy = compute_deltaS_xy(positions)
    # Velocity is computed by scaling displacement with dt: shape (N, 2, T-1)
    vel_xy = compute_vel_xy(ds_xy, dt)

    # -----------------------------
    # 1. Compute Heading
    # -----------------------------
    heading = compute_heading(vel_xy)
    print("\nheading:", heading)
    # Expected shape: (N, T-1)
    assert heading.shape == (N, T - 1), f"Expected heading shape {(N, T - 1)}, got {heading.shape}"

    # -----------------------------
    # 2. Compute Angular Velocity
    # -----------------------------
    ang_vel = compute_angular_velocity(heading, dt)
    print("\nangular velocity:", ang_vel)
    # Expected shape: (N, (T-1)-1) i.e., (N, T-2)
    assert ang_vel.shape == (N, T - 2), f"Expected angular velocity shape {(N, T - 2)}, got {ang_vel.shape}"

    # -----------------------------
    # 3. Compute Turning Angle
    # -----------------------------
    turning_angle = compute_turning_angle(positions)
    print("\nturning angle:", turning_angle)
    # Expected shape: (N, T-2)
    assert turning_angle.shape == (N, T - 2), f"Expected turning angle shape {(N, T - 2)}, got {turning_angle.shape}"
