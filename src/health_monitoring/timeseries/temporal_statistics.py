import numpy as np
from scipy.stats import circmean, circstd


# ====================================
# Temporal Statistical Features
# ====================================


def compute_temporal_linear_statistics(time_series: np.ndarray) -> np.ndarray:
    """
    Compute overall summary statistics over the time axis for each object.

    The computed statistics include:
      - mean
      - median
      - standard deviation
      - minimum
      - maximum
      - iqr

    Parameters
    ----------
    time_series : np.ndarray
        Array of shape (N, T) where N is the number of objects and T is the number
        of time steps.

    Returns
    -------
    stats : np.ndarray
        Array of shape (N, 6) where N is the number of objects and 6 are the summary statistics:
        Dictionary with keys 'mean', 'median', 'variance', 'std', 'min', 'max', 'IQR'.
    """
    assert len(time_series.shape) == 2, f"expected timeseries with shape (N,T). Got shape {time_series.shape}"

    mean = np.mean(time_series, axis=1)
    median = np.median(time_series, axis=1)
    # variance = np.var(time_series, axis=1)
    std = np.std(time_series, axis=1)
    minim = np.min(time_series, axis=1)
    maxim = np.max(time_series, axis=1)
    percentile25 = np.percentile(time_series, 25, axis=1)
    percentile75 = np.percentile(time_series, 75, axis=1)
    iqr = percentile75 - percentile25

    # Stack all statistics column-wise to create (N, 6) array
    return np.column_stack([mean, median, std, minim, maxim, iqr])


def compute_temporal_circular_statistics(angle_array: np.ndarray) -> np.ndarray:
    c_mean = circmean(angle_array, axis=1, high=np.pi, low=-np.pi)
    c_std = circstd(angle_array, axis=1, high=np.pi, low=-np.pi)
    # Stack all statistics column-wise to create (N, 2) array
    return np.stack((c_mean, c_std), axis=1)


def compute_average_timeseries(values: np.ndarray) -> np.ndarray:
    """
    Compute the average time series across multiple objects.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (N, [C], T), where N is the number of objects and T is the number of time steps.
        Each row represents the time series for one object.

    Returns
    -------
    averaged_ts : np.ndarray
        Array of shape ([C], T,) representing the average value across all objects at each time step.
    """

    # Compute mean along the object axis (axis=0)
    averaged_ts = np.mean(values, axis=0)

    return averaged_ts


def compute_median_timeseries(values: np.ndarray) -> np.ndarray:
    """
    Compute the median time series across multiple objects.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (N, [C], T), where N is the number of objects and T is the number of time steps.
        Each row represents the time series for one object.

    Returns
    -------
    averaged_ts : np.ndarray
        Array of shape ([C], T,) representing the median value across all objects at each time step.
    """

    # Compute median along the object axis (axis=0)
    median_ts = np.median(values, axis=0)

    return median_ts


def compute_circular_average_timeseries(values: np.ndarray) -> np.ndarray:
    """
    Compute the circular average time series across multiple objects.

    Parameters
    ----------
    values : np.ndarray
        Array of shape (N, [C], T), where N is the number of objects and T is the number of time steps.
        Each row represents the angle time series for one object (in radians).

    Returns
    -------
    averaged_ts : np.ndarray
        Array of shape ([C], T,) representing the circular average at each time step.
    """

    # Compute circular mean across objects (axis=0)
    averaged_ts = circmean(values, axis=0, high=np.pi, low=-np.pi)

    return averaged_ts


# ==============================
# Example Usage
# ==============================
if __name__ == "__main__":
    N = 3
    T = 10
    # For demonstration, create a dummy time series (e.g., speed or any other metric)
    dummy_time_series = np.random.rand(N, T)  # shape (N, T)

    overall_stats = compute_temporal_linear_statistics(dummy_time_series)
    print("\nOverall Temporal Statistics:")
    print(overall_stats)
