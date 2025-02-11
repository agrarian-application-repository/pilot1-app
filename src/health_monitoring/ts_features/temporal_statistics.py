import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ====================================
# Temporal Statistical Features
# ====================================


def compute_temporal_statistics(time_series: np.ndarray) -> dict:
    """
    Compute overall summary statistics over the time axis for each object.

    The computed statistics include:
      - mean
      - median
      - variance
      - standard deviation
      - minimum
      - maximum
      - 25th percentile
      - 75th percentile

    Parameters
    ----------
    time_series : np.ndarray
        Array of shape (N, T) where N is the number of objects and T is the number
        of time steps.

    Returns
    -------
    stats : dict
        Dictionary with keys 'mean', 'median', 'variance', 'std', 'min', 'max',
        '25th_percentile', and '75th_percentile'. Each value is an array of shape (N,)
        representing the statistic computed over the time axis for each object.
    """
    stats = {
        'mean': np.mean(time_series, axis=1),
        'median': np.median(time_series, axis=1),
        'variance': np.var(time_series, axis=1),
        'std': np.std(time_series, axis=1),
        'min': np.min(time_series, axis=1),
        'max': np.max(time_series, axis=1),
        '25th_percentile': np.percentile(time_series, 25, axis=1),
        '75th_percentile': np.percentile(time_series, 75, axis=1)
    }
    return stats


def compute_sliding_window_statistics(time_series: np.ndarray, window_size: int, step: int = 1) -> dict:
    """
    Compute sliding window statistics over the time axis for each object.

    For each sliding window of a specified size, the function computes the following
    statistics:
      - mean
      - median
      - variance
      - standard deviation
      - minimum
      - maximum
      - 25th percentile
      - 75th percentile

    Parameters
    ----------
    time_series : np.ndarray
        Array of shape (N, T) representing the time series data for N objects.
    window_size : int
        The number of time steps to include in each sliding window.
    step : int, optional
        The step size between consecutive windows (default is 1).

    Returns
    -------
    stats : dict
        Dictionary with keys corresponding to each statistic. Each value is an array of
        shape (N, number_of_windows) representing the statistic computed over each window.
    """
    # Create sliding windows along the time axis.
    # windows shape will be (N, T - window_size + 1, window_size)
    windows = sliding_window_view(time_series, window_shape=window_size, axis=1)
    # Apply stepping (if step > 1)
    windows = windows[:, ::step, :]

    stats = {
        'mean': np.mean(windows, axis=2),
        'median': np.median(windows, axis=2),
        'variance': np.var(windows, axis=2),
        'std': np.std(windows, axis=2),
        'min': np.min(windows, axis=2),
        'max': np.max(windows, axis=2),
        '25th_percentile': np.percentile(windows, 25, axis=2),
        '75th_percentile': np.percentile(windows, 75, axis=2)
    }
    return stats


# ==============================
# Example Usage
# ==============================
if __name__ == "__main__":
    # Assume we have 3 objects, positions in 2D space (normalized between 0 and 1), and 10 time steps.
    N = 3
    T = 10
    positions = np.random.rand(N, 2, T)

    # ----- Temporal Statistical Features -----
    # For demonstration, create a dummy time series (e.g., speed or any other metric)
    dummy_time_series = np.random.rand(N, T)  # shape (N, T)

    overall_stats = compute_temporal_statistics(dummy_time_series)
    print("\nOverall Temporal Statistics:")
    for key, value in overall_stats.items():
        print(f"{key}: {value}")

    # Compute sliding window statistics with a window size of 3 time steps
    window_stats = compute_sliding_window_statistics(dummy_time_series, window_size=3, step=1)
    print("\nSliding Window Statistics (mean) shape:", window_stats['mean'].shape)  # (N, T - window_size + 1)
