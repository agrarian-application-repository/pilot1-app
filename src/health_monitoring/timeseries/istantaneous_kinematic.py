import numpy as np


# =============================================================================
# General Purpose Functions
# =============================================================================

def compute_delta_over_timestep(array: np.ndarray) -> np.ndarray:
    """
    Compute the difference between consecutive time steps along the time axis.

    Parameters
    ----------
    array : np.ndarray
        Input array of shape (N, C, T) where:
          - N is the number of objects,
          - C is the number of coordinates (e.g. 2 for x and y),
          - T is the number of time steps.

    Returns
    -------
    delta_array : np.ndarray
        Array of shape (N, C, T-1) representing the differences over each coordinate between consecutive time steps.
    """
    return np.diff(array, axis=2)


def compute_magnitude(vectors: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean norm (magnitude) of vectors along the coordinate axis.

    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (N, C, T) where C is the number of coordinates (e.g. 2 for x and y).

    Returns
    -------
    magnitude : np.ndarray
        Array of shape (N, T) containing the Euclidean norm of the vectors.
    """
    return np.linalg.norm(vectors, axis=1)


def compute_direction(vectors: np.ndarray) -> np.ndarray:
    """
    Compute the direction (angle) of 2D vectors using arctan2.

    The function returns the angle in radians relative to the positive x-axis.

    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (N, 2, T) representing the 2D vectors for N objects over T time steps.
        The first coordinate (index 0) is the x-component and the second (index 1) is the y-component.

    Returns
    -------
    direction : np.ndarray
        Array of shape (N, T) containing the direction (angle in radians) for each vector.
    """
    return np.arctan2(vectors[:, 1, :], vectors[:, 0, :])


def compute_time_derivative(delta_array: np.ndarray, dt: float) -> np.ndarray:
    """
    Scale the array by dividing by the time interval dt.

    Parameters
    ----------
    delta_array : np.ndarray
        Input array (e.g., differences computed over time).
    dt : float
        The time interval between consecutive measurements.

    Returns
    -------
    scaled_data : np.ndarray
        The input array divided by dt.
    """
    return delta_array / dt


def encode_direction(direction_angles: np.ndarray) -> np.ndarray:
    """
    Encode direction angles into their sine and cosine components.

    This function converts a (N, T) array of direction angles into a (N, 2, T) array,
    where the first row represents sine values and the second row represents cosine values.

    Parameters
    ----------
    direction_angles : np.ndarray
        Array of shape (N, T) containing direction angles in radians.

    Returns
    -------
    encoded_directions : np.ndarray
        Array of shape (N, 2, T) where:
        - encoded_directions[:, 0, :] contains sin(direction_angles)
        - encoded_directions[:, 1, :] contains cos(direction_angles)
    """
    sin_values = np.sin(direction_angles)  # Compute sine values
    cos_values = np.cos(direction_angles)  # Compute cosine values

    # Stack along the second axis to form a (N, 2, T) array
    encoded_directions = np.stack((sin_values, cos_values), axis=1)

    return encoded_directions


# =============================================================================
# Kinematics Functions (implemented in terms of the general functions)
# =============================================================================

# --- Displacement ---
def compute_deltaS_xy(pos_xy: np.ndarray) -> np.ndarray:
    """
    Compute displacement vectors (deltaS) from positions.

    Parameters
    ----------
    pos_xy : np.ndarray
        Array of shape (N, 2, T) representing positions.

    Returns
    -------
    ds_xy : np.ndarray
        Array of shape (N, 2, T-1) representing the difference between consecutive positions.
    """
    return compute_delta_over_timestep(pos_xy)


def compute_deltaS_magnitude(ds_xy: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of the displacement vectors.

    Parameters
    ----------
    ds_xy : np.ndarray
        Array of shape (N, 2, T-1) representing displacement vectors.

    Returns
    -------
    ds : np.ndarray
        Array of shape (N, T-1) with the Euclidean norm of each displacement vector.
    """
    return compute_magnitude(ds_xy)


def compute_deltaS_direction(ds_xy: np.ndarray) -> np.ndarray:
    """
    Compute the direction (angle) of displacement vectors.

    Parameters
    ----------
    ds_xy : np.ndarray
        Array of shape (N, 2, T-1) representing displacement vectors.

    Returns
    -------
    direction : np.ndarray
        Array of shape (N, T-1) with the direction (angle in radians) of each displacement vector.
    """
    return compute_direction(ds_xy)


# --- Velocity ---
def compute_vel_xy(ds_xy: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute velocity vectors from displacement vectors.

    Parameters
    ----------
    ds_xy : np.ndarray
        Array of shape (N, 2, T-1) representing displacement vectors.
    dt : float
        Time interval between measurements.

    Returns
    -------
    vel_xy : np.ndarray
        Array of shape (N, 2, T-1) representing velocity vectors.
    """
    return compute_time_derivative(ds_xy, dt)


def compute_vel_magnitude(vel_xy: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of the velocity vectors (i.e. speed).

    Parameters
    ----------
    vel_xy : np.ndarray
        Array of shape (N, 2, T-1) representing velocity vectors.

    Returns
    -------
    vel : np.ndarray
        Array of shape (N, T-1) with the Euclidean norm of each velocity vector.
    """
    return compute_magnitude(vel_xy)


def compute_vel_direction(vel_xy: np.ndarray) -> np.ndarray:
    """
    Compute the direction (angle) of velocity vectors.

    Parameters
    ----------
    vel_xy : np.ndarray
        Array of shape (N, 2, T-1) representing velocity vectors.

    Returns
    -------
    direction : np.ndarray
        Array of shape (N, T-1) with the direction (angle in radians) of each velocity vector.
    """
    return compute_direction(vel_xy)


def compute_deltaV_xy(vel_xy: np.ndarray) -> np.ndarray:
    """
    Compute the change in velocity vectors (deltaV).

    Parameters
    ----------
    vel_xy : np.ndarray
        Array of shape (N, 2, T-1) representing velocity vectors.

    Returns
    -------
    dv_xy : np.ndarray
        Array of shape (N, 2, T-2) representing the difference between consecutive velocity vectors.
    """
    return compute_delta_over_timestep(vel_xy)


def compute_deltaV_magnitude(dv_xy: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of the change in velocity vectors.

    Parameters
    ----------
    dv_xy : np.ndarray
        Array of shape (N, 2, T-2) representing delta_array velocity vectors.

    Returns
    -------
    dv : np.ndarray
        Array of shape (N, T-2) with the Euclidean norm of each delta_array velocity vector.
    """
    return compute_magnitude(dv_xy)


def compute_deltaV_direction(dv_xy: np.ndarray) -> np.ndarray:
    """
    Compute the direction (angle) of delta_array velocity (change in velocity) vectors.

    Parameters
    ----------
    dv_xy : np.ndarray
        Array of shape (N, 2, T-2) representing delta_array velocity vectors.

    Returns
    -------
    direction : np.ndarray
        Array of shape (N, T-2) with the direction (angle in radians) of each delta_array velocity vector.
    """
    return compute_direction(dv_xy)


# --- Acceleration ---
def compute_acc_xy(dv_xy: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute acceleration vectors from delta_array velocity vectors.

    Parameters
    ----------
    dv_xy : np.ndarray
        Array of shape (N, 2, T-2) representing delta_array velocity vectors.
    dt : float
        Time interval between consecutive velocity measurements.

    Returns
    -------
    acc_xy : np.ndarray
        Array of shape (N, 2, T-2) representing acceleration vectors.
    """
    return compute_time_derivative(dv_xy, dt)


def compute_acc_magnitude(acc_xy: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of the acceleration vectors.

    Parameters
    ----------
    acc_xy : np.ndarray
        Array of shape (N, 2, T-2) representing acceleration vectors.

    Returns
    -------
    acc : np.ndarray
        Array of shape (N, T-2) with the Euclidean norm of each acceleration vector.
    """
    return compute_magnitude(acc_xy)


def compute_acc_direction(acc_xy: np.ndarray) -> np.ndarray:
    """
    Compute the direction (angle) of acceleration vectors.

    Parameters
    ----------
    acc_xy : np.ndarray
        Array of shape (N, 2, T-2) representing acceleration vectors.

    Returns
    -------
    direction : np.ndarray
        Array of shape (N, T-2) with the direction (angle in radians) of each acceleration vector.
    """
    return compute_direction(acc_xy)


def compute_deltaA_xy(acc_xy: np.ndarray) -> np.ndarray:
    """
    Compute the change in acceleration vectors (deltaA).

    Parameters
    ----------
    acc_xy : np.ndarray
        Array of shape (N, 2, T-2) representing acceleration vectors.

    Returns
    -------
    da_xy : np.ndarray
        Array of shape (N, 2, T-3) representing the difference between consecutive acceleration vectors.
    """
    return compute_delta_over_timestep(acc_xy)


def compute_deltaA_magnitude(da_xy: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of the change in acceleration vectors.

    Parameters
    ----------
    da_xy : np.ndarray
        Array of shape (N, 2, T-3) representing delta_array acceleration vectors.

    Returns
    -------
    da : np.ndarray
        Array of shape (N, T-3) with the Euclidean norm of each delta_array acceleration vector.
    """
    return compute_magnitude(da_xy)


def compute_deltaA_direction(da_xy: np.ndarray) -> np.ndarray:
    """
    Compute the direction (angle) of delta_array acceleration vectors.

    Parameters
    ----------
    da_xy : np.ndarray
        Array of shape (N, 2, T-3) representing delta_array acceleration vectors.

    Returns
    -------
    direction : np.ndarray
        Array of shape (N, T-3) with the direction (angle in radians) of each delta_array acceleration vector.
    """
    return compute_direction(da_xy)


# --- Jerk ---
def compute_jerk_xy(da_xy: np.ndarray, dt: float) -> np.ndarray:
    """
    Compute jerk vectors from delta_array acceleration vectors.

    Parameters
    ----------
    da_xy : np.ndarray
        Array of shape (N, 2, T-3) representing delta_array acceleration vectors.
    dt : float
        Time interval between consecutive acceleration measurements.

    Returns
    -------
    jerk_xy : np.ndarray
        Array of shape (N, 2, T-3) representing jerk vectors.
    """
    return compute_time_derivative(da_xy, dt)


def compute_jerk_magnitude(jerk_xy: np.ndarray) -> np.ndarray:
    """
    Compute the magnitude of the jerk vectors.

    Parameters
    ----------
    jerk_xy : np.ndarray
        Array of shape (N, 2, T-3) representing jerk vectors.

    Returns
    -------
    jerk : np.ndarray
        Array of shape (N, T-3) with the Euclidean norm of each jerk vector.
    """
    return compute_magnitude(jerk_xy)


def compute_jerk_direction(jerk_xy: np.ndarray) -> np.ndarray:
    """
    Compute the direction (angle) of jerk vectors.

    Parameters
    ----------
    jerk_xy : np.ndarray
        Array of shape (N, 2, T-3) representing jerk vectors.

    Returns
    -------
    direction : np.ndarray
        Array of shape (N, T-3) with the direction (angle in radians) of each jerk vector.
    """
    return compute_direction(jerk_xy)


# =============================================================================
# Main: Testing the Functions
# =============================================================================

if __name__ == "__main__":
    # Create a dummy dataset:
    # Let's assume we have 3 objects, positions in 2D space, and 10 time steps.
    N = 3
    T = 10
    dt = 2.0
    # For reproducibility, generate random positions between 0 and 1.
    positions = np.random.rand(N, 2, T)
    print("positions:", positions)

    # Compute displacement vectors (deltaS_xy)
    ds_xy = compute_deltaS_xy(positions)
    print("\ndeltaS_xy (ds_xy):", ds_xy)
    assert ds_xy.shape == (N, 2, T - 1), f"Expected ds_xy shape {(N, 2, T - 1)}, got {ds_xy.shape}"

    # Compute displacement magnitude (ds)
    ds = compute_deltaS_magnitude(ds_xy)
    print("\ndeltaS magnitude (ds):", ds)
    assert ds.shape == (N, T - 1), f"Expected ds shape {(N, T - 1)}, got {ds.shape}"

    # Compute displacement direction (ds_direction)
    ds_direction = compute_deltaS_direction(ds_xy)
    print("ds_direction:", ds_direction)
    assert ds_direction.shape == (N, T - 1), f"Expected shape {(N, T -    1)}, got {ds_direction.shape}"

    # Compute velocity vectors (vel_xy)
    vel_xy = compute_vel_xy(ds_xy, dt)
    print("\nvelocity vectors (vel_xy):", vel_xy)
    assert vel_xy.shape == (N, 2, T - 1), f"Expected vel_xy shape {(N, 2, T - 1)}, got {vel_xy.shape}"

    # Compute velocity magnitude (vel)
    vel = compute_vel_magnitude(vel_xy)
    print("\nvelocity magnitude (vel):", vel)
    assert vel.shape == (N, T - 1), f"Expected vel shape {(N, T - 1)}, got {vel.shape}"

    # Compute velocity direction (vel_direction)
    vel_direction = compute_vel_direction(vel_xy)
    print("vel_direction:", vel_direction)
    assert vel_direction.shape == (N, T - 1), f"Expected shape {(N, T - 1)}, got {vel_direction.shape}"

    # Compute delta_array velocity vectors (dv_xy)
    dv_xy = compute_deltaV_xy(vel_xy)
    print("\ndelta_array velocity vectors (dv_xy):", dv_xy)
    assert dv_xy.shape == (N, 2, T - 2), f"Expected dv_xy shape {(N, 2, T - 2)}, got {dv_xy.shape}"

    # Compute delta_array velocity magnitude (dv)
    dv = compute_deltaV_magnitude(dv_xy)
    print("\ndelta_array velocity magnitude (dv):", dv)
    assert dv.shape == (N, T - 2), f"Expected dv shape {(N, T - 2)}, got {dv.shape}"

    # Compute delta_array velocity direction (dv_direction)
    dv_direction = compute_deltaV_direction(dv_xy)
    print("dv_direction:", dv_direction)
    assert dv_direction.shape == (N, T - 2), f"Expected shape {(N, T - 2)}, got {dv_direction.shape}"

    # Compute acceleration vectors (acc_xy)
    acc_xy = compute_acc_xy(dv_xy, dt)
    print("\nacceleration vectors (acc_xy):", acc_xy)
    assert acc_xy.shape == (N, 2, T - 2), f"Expected acc_xy shape {(N, 2, T - 2)}, got {acc_xy.shape}"

    # Compute acceleration magnitude (acc)
    acc = compute_acc_magnitude(acc_xy)
    print("\nacceleration magnitude (acc):", acc)
    assert acc.shape == (N, T - 2), f"Expected acc shape {(N, T - 2)}, got {acc.shape}"

    # Compute acceleration direction (acc_direction)
    acc_direction = compute_acc_direction(acc_xy)
    print("acc_direction:", acc_direction)
    assert acc_direction.shape == (N, T - 2), f"Expected shape {(N, T - 2)}, got {acc_direction.shape}"

    # Compute delta_array acceleration vectors (da_xy)
    da_xy = compute_deltaA_xy(acc_xy)
    print("\ndelta_array acceleration vectors (da_xy):", da_xy)
    assert da_xy.shape == (N, 2, T - 3), f"Expected da_xy shape {(N, 2, T - 3)}, got {da_xy.shape}"

    # Compute delta_array acceleration magnitude (da)
    da = compute_deltaA_magnitude(da_xy)
    print("\ndelta_array acceleration magnitude (da):", da)
    assert da.shape == (N, T - 3), f"Expected da shape {(N, T - 3)}, got {da.shape}"

    # Compute delta_array acceleration direction (da_direction)
    da_direction = compute_deltaA_direction(da_xy)
    print("da_direction:", da_direction)
    assert da_direction.shape == (N, T - 3), f"Expected shape {(N, T - 3)}, got {da_direction.shape}"

    # Compute jerk vectors (jerk_xy)
    jerk_xy = compute_jerk_xy(da_xy, dt)
    print("\njerk vectors (jerk_xy):", jerk_xy)
    assert jerk_xy.shape == (N, 2, T - 3), f"Expected jerk_xy shape {(N, 2, T - 3)}, got {jerk_xy.shape}"

    # Compute jerk magnitude (jerk)
    jerk = compute_jerk_magnitude(jerk_xy)
    print("\njerk magnitude (jerk):", jerk)
    assert jerk.shape == (N, T - 3), f"Expected jerk shape {(N, T - 3)}, got {jerk.shape}"

    # Compute jerk direction (jerk_direction)
    jerk_direction = compute_jerk_direction(jerk_xy)
    print("jerk_direction:", jerk_direction)
    assert jerk_direction.shape == (N, T - 3), f"Expected shape {(N, T - 3)}, got {jerk_direction.shape}"
