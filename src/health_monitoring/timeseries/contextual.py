import numpy as np


# ----------------------------------------------------
# Contextual or Derived Features
# ----------------------------------------------------

import numpy as np


def compute_centroid_timeseries(positions: np.ndarray) -> np.ndarray:
    """
    Computes the time series of the centroid from a time series of (x, y) positions
    of multiple objects.

    Parameters:
    -----------
    positions : np.ndarray
        A NumPy array of shape (N, 2, T) where:
            - N is the number of objects
            - 2 corresponds to x and y coordinates
            - T is the number of time steps

    Returns:
    --------
    centroid_ts : np.ndarray
        A NumPy array of shape (2, T) representing the (x, y) coordinates of the
        centroid at each time step.
    """

    # Compute the mean across the object dimension (axis=0)
    centroid_ts = np.mean(positions, axis=0)  # shape will be (2, T)

    return centroid_ts


def compute_relative_position(positions: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
    """
    Compute the relative position of each object with respect to a given reference point.

    The relative position is calculated as the difference between each object's position
    and the reference point.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing the positions of N objects over T time steps.
    reference_point : np.ndarray
        Array of shape (2,) representing the [x, y] coordinates of the reference point.

    Returns
    -------
    relative_positions : np.ndarray
        Array of shape (N, 2, T) representing the positions relative to the reference point.
    """
    # Ensure reference_point is broadcastable to (N, 2, T)
    relative_positions = positions - reference_point.reshape(1, 2, 1)
    return relative_positions


def compute_distance_to_fixed_reference(positions: np.ndarray, reference_point: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance from each object to a given reference point at each time step.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing the positions of N objects over T time steps.
    reference_point : np.ndarray
        Array of shape (2,) representing the [x, y] coordinates of the reference point.

    Returns
    -------
    distances : np.ndarray
        Array of shape (N, T) containing the Euclidean distances from each object to the reference point.
    """
    # Compute the relative positions first
    relative_positions = compute_relative_position(positions, reference_point)
    # Compute the Euclidean norm along the coordinate axis (axis=1)
    distances = np.linalg.norm(relative_positions, axis=1)
    return distances


def compute_distance_to_moving_reference(positions: np.ndarray, reference_ts: np.ndarray) -> np.ndarray:
    """
    Compute the Euclidean distance from each object to a moving reference point at each time step.

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing the (x, y) positions of N objects over T time steps.
    reference_ts : np.ndarray
        Array of shape (2, T) representing the (x, y) coordinates of the moving reference at each time step.

    Returns
    -------
    distances : np.ndarray
        Array of shape (N, T) containing the Euclidean distances from each object to the reference point at each time.
    """

    # Compute the relative positions
    relative_positions = positions - reference_ts[np.newaxis, :, :]  # shape (N, 2, T)
    # Compute Euclidean distances along the coordinate axis
    distances = np.linalg.norm(relative_positions, axis=1)  # shape (N, T)

    return distances


def compute_zone_indicators(positions: np.ndarray, x_max: float = 1.0, y_max: float = 1.0, n_rows: int = 2, n_cols: int = 2) -> np.ndarray:
    """
    Compute zone indicators for each object by assigning a zone label based on its position in the space.

    The 2D space is partitioned into a grid of n_rows x n_cols,
    and bounded in [0,x_max] along the columns, and bounded in [0,y_max] along the rows

    Each object's zone is determined by its (x, y) coordinates. The zone label is computed as:

        zone = row_index * n_cols + col_index

    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 2, T) representing the positions (x, y) of N objects over T time steps.
    x_max : float
        Maximum value for the x-axis, used to normalize x coordinates.
    y_max : float
        Maximum value for the y-axis, used to normalize y coordinates.
    n_rows : int, optional
        Number of rows to divide the space into (default is 2).
    n_cols : int, optional
        Number of columns to divide the space into (default is 2).

    Returns
    -------
    zones : np.ndarray
        Array of shape (N, T) containing an integer zone label for each object at each time step.
    """
    N, _, T = positions.shape
    x = positions[:, 0, :]
    y = positions[:, 1, :]

    # Scale x and y according to the provided max values
    col_index = np.clip((x / x_max * n_cols).astype(int), 0, n_cols - 1)
    row_index = np.clip((y / y_max * n_rows).astype(int), 0, n_rows - 1)

    # Compute zone indicator: zone = row_index * n_cols + col_index
    zones = row_index * n_cols + col_index
    return zones


def compute_lagged_features(time_series: np.ndarray, lags: list) -> dict:
    """
    Compute lagged versions of a given time series for each object.

    For each specified lag, this function returns a shifted version of the time series.
    The lagged feature at lag `l` is defined such that the value at time t corresponds to
    the original value at time t - l. The beginning of the series is padded with NaN to maintain
    the same shape.

    Parameters
    ----------
    time_series : np.ndarray
        Array of shape (N, T) representing a time series for N objects over T time steps.
    lags : list of int
        List of positive integer lags for which to compute the lagged features.

    Returns
    -------
    lagged_features : dict
        Dictionary where each key is a lag value (int) and each value is an array of shape (N, T)
        representing the time series shifted by that lag.
    """
    assert len(time_series.shape) == 2, f"expected timeseries with shape (N,T). Got shape {time_series.shape}"

    N, T = time_series.shape
    lagged_features = {}

    for lag in lags:
        # Create an array filled with NaN values to pad the initial lag entries
        lagged = np.full((N, T), np.nan)
        if lag < T:
            # Shift the series: For time indices lag...T, fill with values from indices 0...T-lag
            lagged[:, lag:] = time_series[:, :T - lag]
        lagged_features[lag] = lagged
    return lagged_features


# ----------------------------------------------------
# Example Usage
# ----------------------------------------------------
if __name__ == "__main__":
    # Assume we have 5 objects, with positions in a normalized 2D space ([0,1] for both x and y)
    # and 10 time steps.
    N = 5
    T = 10
    positions = np.random.rand(N, 2, T)

    # 1. Compute Relative Positions to a Reference Point (e.g., center of the space)
    reference_point = np.array([0.5, 0.5])
    relative_positions = compute_relative_position(positions, reference_point)
    print("Relative Positions shape:", relative_positions.shape)  # Expected: (5, 2, 10)

    # 2. Compute Distance to the Reference Point
    distances = compute_distance_to_fixed_reference(positions, reference_point)
    print("Distance to Reference shape:", distances.shape)  # Expected: (5, 10)

    # 3. Compute Zone Indicators (using a 2x2 grid by default)
    zones = compute_zone_indicators(positions, n_rows=2, n_cols=2)
    print("Zone Indicators shape:", zones.shape)  # Expected: (5, 10)
    print("Zone Indicators:\n", zones)

    # 4. Compute Lagged Features for a dummy time series
    # For demonstration, use the x-coordinate time series of the first object.
    dummy_time_series = positions[0, 0, :].reshape(1, T)  # Shape: (1, 10)
    lags = [1, 2, 3]
    lagged_feats = compute_lagged_features(dummy_time_series, lags)
    for lag, feat in lagged_feats.items():
        print(f"\nLag {lag} feature:\n", feat)
