import numpy as np
import pywt  # PyWavelets library

# =====================================================
# Frequency Domain Features: FFT-Based Features
# =====================================================


def compute_fft_features(time_series: np.ndarray, dt: float = 1.0) -> dict:
    """
    Compute frequency domain features from a 1D time series for each object using FFT.

    This function computes the FFT of the input time series and extracts:
      - Dominant frequency (ignoring the DC component)
      - Spectral centroid (weighted average frequency)
      - Spectral bandwidth (weighted standard deviation of frequencies)
      - Total spectral energy (sum of squared magnitudes)

    Parameters
    ----------
    time_series : np.ndarray
        Array of shape (N, T), where N is the number of objects and T is the number
        of time steps.
    dt : float, optional
        Sampling interval between time steps (default is 1.0).

    Returns
    -------
    fft_features : dict
        Dictionary containing:
          'dominant_frequency': np.ndarray of shape (N,)
          'spectral_centroid':  np.ndarray of shape (N,)
          'spectral_bandwidth': np.ndarray of shape (N,)
          'spectral_energy':    np.ndarray of shape (N,)
    """
    assert len(time_series.shape) == 2, f"expected timeseries with shape (N,T). Got shape {time_series.shape}"

    N, T = time_series.shape
    # Compute FFT along the time axis (using rfft for real-valued signals)
    fft_coeffs = np.fft.rfft(time_series, axis=1)
    # Compute the corresponding frequency bins
    freqs = np.fft.rfftfreq(T, d=dt)

    # Compute the magnitude spectrum
    magnitude = np.abs(fft_coeffs)

    # Total spectral energy: sum of squared magnitudes for each object
    spectral_energy = np.sum(magnitude ** 2, axis=1)

    # Spectral centroid: weighted average frequency
    spectral_centroid = np.sum(magnitude * freqs, axis=1) / np.sum(magnitude, axis=1)

    # Spectral bandwidth: weighted standard deviation of the frequencies
    spectral_bandwidth = np.sqrt(
        np.sum(magnitude * (freqs - spectral_centroid[:, None]) ** 2, axis=1) /
        np.sum(magnitude, axis=1)
    )

    # Dominant frequency: frequency corresponding to the maximum magnitude
    # Ignore the DC component (frequency = 0) by setting its magnitude to zero
    mag_no_dc = magnitude.copy()
    mag_no_dc[:, 0] = 0
    dominant_indices = np.argmax(mag_no_dc, axis=1)
    dominant_frequency = freqs[dominant_indices]

    fft_features = {
        'dominant_frequency': dominant_frequency,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_energy': spectral_energy
    }
    return fft_features


# =====================================================
# Frequency Domain Features: Wavelet Energy Features
# =====================================================

def compute_wavelet_energy_features(time_series: np.ndarray, wavelet: str = 'db4', level: int = None) -> dict:
    """
    Compute wavelet energy features for each object's time series using discrete wavelet transform.

    For each object's time series, this function performs a wavelet decomposition and computes
    the energy (sum of squares of coefficients) of the approximation coefficients at the final level
    and the detail coefficients at each decomposition level.

    Parameters
    ----------
    time_series : np.ndarray
        Array of shape (N, T), where N is the number of objects and T is the number of time steps.
    wavelet : str, optional
        Name of the wavelet to use for decomposition (default is 'db4').
    level : int, optional
        Decomposition level. If None, the maximum level possible is used.

    Returns
    -------
    wavelet_features : dict
        Dictionary containing:
          'approximation_energy': np.ndarray of shape (N,)
          'level_1': np.ndarray of shape (N,) (energy of detail coefficients at level 1)
          'level_2': np.ndarray of shape (N,) (energy of detail coefficients at level 2)
          ... (one key per detail level)
    """
    assert len(time_series.shape) == 2, f"expected timeseries with shape (N,T). Got shape {time_series.shape}"

    N, T = time_series.shape
    approx_energy_list = []
    detail_energy_lists = []  # This will be a list of lists; each inner list holds energies for all levels for one object

    # Process each object's time series individually
    for i in range(N):
        # Decompose the time series for object i
        coeffs = pywt.wavedec(time_series[i, :], wavelet, level=level)
        # coeffs[0] contains the approximation coefficients at the final level
        approx = coeffs[0]
        energy_approx = np.sum(np.square(approx))
        approx_energy_list.append(energy_approx)

        # coeffs[1:] are the detail coefficients at each level (level 1, 2, ...)
        detail_energies = []
        for detail_coeff in coeffs[1:]:
            energy_detail = np.sum(np.square(detail_coeff))
            detail_energies.append(energy_detail)
        detail_energy_lists.append(detail_energies)

    # Convert the approximation energies to a NumPy array
    approx_energy_array = np.array(approx_energy_list)  # Shape: (N,)

    # Assume all objects yield the same number of detail levels
    num_levels = len(detail_energy_lists[0])
    detail_energy_arrays = {}
    for lvl in range(num_levels):
        # Gather energy for level lvl+1 for each object
        detail_energy_arrays[f'level_{lvl + 1}'] = np.array(
            [detail_energy_lists[i][lvl] for i in range(N)]
        )

    # Combine the features into a single dictionary
    wavelet_features = {'approximation_energy': approx_energy_array}
    wavelet_features.update(detail_energy_arrays)

    return wavelet_features


# =====================================================
# Example Usage
# =====================================================

if __name__ == "__main__":
    # Create a dummy time series dataset.
    # For example, assume we have 3 objects, each with 100 time steps.
    N = 3
    T = 100
    dt = 0.01  # Sampling interval (seconds)

    # Create a time vector
    t = np.linspace(0, (T - 1) * dt, T)

    # Generate example signals with distinct dominant frequencies for each object.
    time_series = np.zeros((N, T))
    time_series[0, :] = np.sin(2 * np.pi * 5 * t)  # Object 0: 5 Hz sine wave
    time_series[1, :] = np.sin(2 * np.pi * 10 * t)  # Object 1: 10 Hz sine wave
    time_series[2, :] = np.sin(2 * np.pi * 20 * t)  # Object 2: 20 Hz sine wave

    # --------------------------
    # FFT-Based Features
    # --------------------------
    fft_feats = compute_fft_features(time_series, dt=dt)
    print("FFT-Based Frequency Domain Features:")
    for key, value in fft_feats.items():
        print(f"{key}: {value}")

    # --------------------------
    # Wavelet Energy Features
    # --------------------------
    # Ensure that PyWavelets is installed (pip install PyWavelets)
    wavelet_feats = compute_wavelet_energy_features(time_series, wavelet='db4')
    print("\nWavelet Energy Features:")
    for key, value in wavelet_feats.items():
        print(f"{key}: {value}")
