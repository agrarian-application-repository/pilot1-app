import numpy as np

def compute_displacement(positions: np.ndarray) -> np.ndarray:
    """
    Compute displacement vectors (differences between consecutive positions)
    for each object over time.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) where:
          - N is the number of objects,
          - 2 corresponds to the x and y coordinates (normalized between 0 and 1),
          - T is the number of time steps.

    Returns
    -------
    displacement : np.ndarray
        Array of shape (N, 2, T-1) representing the displacement vectors between
        consecutive time steps for each object.
    """
    displacement = np.diff(positions, axis=2)
    return displacement


def compute_velocity(positions: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute velocity vectors for each object by calculating the displacement
    over time. Assumes constant time intervals defined by dt.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing the [x, y] positions over time.
    dt : float, optional
        Time interval between consecutive samples (default is 1.0).

    Returns
    -------
    velocity : np.ndarray
        Array of shape (N, 2, T-1) representing the velocity vectors computed
        as displacement divided by dt.
    """
    displacement = compute_displacement(positions)
    velocity = displacement / dt
    return velocity


def compute_speed(velocity: np.ndarray) -> np.ndarray:
    """
    Compute the speed (magnitude of the velocity vector) for each object.

    Parameters
    ----------
    velocity : np.ndarray
        Array of shape (N, 2, T) representing the velocity vectors.

    Returns
    -------
    speed : np.ndarray
        Array of shape (N, T) representing the speed (Euclidean norm of velocity)
        for each object over time.
    """
    # Compute Euclidean norm over the 2D (x,y) components (axis=1)
    speed = np.linalg.norm(velocity, axis=1)
    return speed


def compute_acceleration(velocity: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute acceleration vectors for each object by taking the time derivative
    (difference) of the velocity vectors.

    Parameters
    ----------
    velocity : np.ndarray
        Array of shape (N, 2, T) representing velocity vectors.
    dt : float, optional
        Time interval between consecutive velocity measurements (default is 1.0).

    Returns
    -------
    acceleration : np.ndarray
        Array of shape (N, 2, T-1) representing the acceleration vectors computed
        as the difference between consecutive velocity vectors divided by dt.
    """
    acceleration = np.diff(velocity, axis=2) / dt
    return acceleration


def compute_acceleration_magnitude(acceleration: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude (Euclidean norm) of the acceleration vectors.

    Parameters
    ----------
    acceleration : np.ndarray
        Array of shape (N, 2, T) representing the acceleration vectors.

    Returns
    -------
    acc_magnitude : np.ndarray
        Array of shape (N, T) representing the magnitude of acceleration for each
        object over time.
    """
    acc_magnitude = np.linalg.norm(acceleration, axis=1)
    return acc_magnitude


def compute_jerk(acceleration: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """
    Compute the jerk (rate of change of acceleration) for each object.

    Parameters
    ----------
    acceleration : np.ndarray
        Array of shape (N, 2, T) representing the acceleration vectors.
    dt : float, optional
        Time interval between consecutive acceleration measurements (default is 1.0).

    Returns
    -------
    jerk : np.ndarray
        Array of shape (N, 2, T-1) representing the jerk vectors computed as the
        difference between consecutive acceleration vectors divided by dt.
    """
    jerk = np.diff(acceleration, axis=2) / dt
    return jerk


# Example usage:
if __name__ == "__main__":
    # Create a dummy dataset:
    # Let's assume we have 3 objects, positions in 2D space, and 10 time steps.
    N = 3
    T = 10
    # For reproducibility, generate random positions between 0 and 1.
    positions = np.random.rand(N, 2, T)

    # Compute instantaneous kinematics
    disp = compute_displacement(positions)
    vel = compute_velocity(positions, dt=1.0)
    speed = compute_speed(vel)
    acc = compute_acceleration(vel, dt=1.0)
    acc_magnitude = compute_acceleration_magnitude(acc)
    jerk = compute_jerk(acc, dt=1.0)

    # Print shapes of computed features
    print("Positions shape:", positions.shape)         # (N, 2, T)
    print("Displacement shape:", disp.shape)             # (N, 2, T-1)
    print("Velocity shape:", vel.shape)                  # (N, 2, T-1)
    print("Speed shape:", speed.shape)                   # (N, T-1)
    print("Acceleration shape:", acc.shape)              # (N, 2, T-2)
    print("Acceleration Magnitude shape:", acc_magnitude.shape)  # (N, T-2)
    print("Jerk shape:", jerk.shape)                     # (N, 2, T-3)
