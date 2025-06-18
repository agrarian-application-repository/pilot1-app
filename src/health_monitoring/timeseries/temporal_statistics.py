import numpy as np
from typing import Union
from scipy.stats import circmean, circstd


# ====================================
# Temporal Statistical Features
# ====================================


def compute_temporal_linear_statistics(time_series: np.ndarray, return_array: bool = True) -> Union[dict, np.ndarray]:
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

    return_array: bool

    Returns
    -------
    stats : dict
        Dictionary with keys 'mean', 'median', 'variance', 'std', 'min', 'max', 'IQR'.
        Each value is an array of shape (N,)
        representing the statistic computed over the time axis for each object.
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

    if return_array:
        # Stack all statistics column-wise to create (N, 6) array
        return np.column_stack([mean, median, std, minim, maxim, iqr])

    stats = {
        'mean': mean,
        'median': median,
        'std': std,
        'min': minim,
        'max': maxim,
        'iqr': iqr
    }
    return stats


def compute_temporal_circular_statistics(angle_array: np.ndarray, return_array: bool = True) -> Union[dict, np.ndarray]:
    c_mean = circmean(angle_array, axis=1, high=np.pi, low=-np.pi)
    c_std = circstd(angle_array, axis=1, high=np.pi, low=-np.pi)

    if return_array:
        return np.stack((c_mean, c_std), axis=1)

    stats = {
        'circular_mean': c_mean,
        'circular_std': c_std,
    }
    return stats


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
    for key, value in overall_stats.items():
        print(f"{key}: {value}")
