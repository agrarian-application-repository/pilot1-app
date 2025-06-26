from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from scipy.stats import zscore
from scipy.spatial.distance import mahalanobis

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import math


def save_feature_distributions(data: np.ndarray, filename: str = "feature_distributions.png"):
    """
    Plot histograms of each feature in a (N, F) NumPy array and save to a file.

    Parameters:
        data (np.ndarray): Input data of shape (N, F)
        filename (str): Output filename (with extension, e.g., .png or .pdf)
    """
    N, F = data.shape
    n_cols = 4  # number of subplots per row
    n_rows = math.ceil(F / n_cols)

    titles = [
        "ds_m_mean",
        "ds_m_median",
        "ds_m_std",
        "ds_m_min",
        "ds_m_max",
        "ds_m_iqr",
        "v_m_mean",
        "v_m_median",
        "v_m_std",
        "v_m_min",
        "v_m_max",
        "v_m_iqr",
        "a_m_mean",
        "a_m_median",
        "a_m_std",
        "a_m_min",
        "a_m_max",
        "a_m_iqr",
        "ds_angle_circ_mean",
        "ds_angle_circ_std",
        "tot_disp",
        "top_path_eff",
        "dist_centroid_mean",
        "dist_centroid_median",
        "dist_centroid_std",
        "dist_centroid_min",
        "dist_centroid_max",
        "dist_centroid_iqr",
        "local_density_mean",
        "local_density_median",
        "local_density_std",
        "local_density_min",
        "local_density_max",
        "local_density_iqr",
        "avg_knn_mean",
        "avg_knn_median",
        "avg_knn_std",
        "avg_knn_min",
        "avg_knn_max",
        "avg_knn_iqr",
        "diff_v_m_mean",
        "diff_v_m_median",
        "diff_v_m_std",
        "diff_v_m_min",
        "diff_v_m_max",
        "diff_v_m_iqr",
        "diff_a_m_mean",
        "diff_a_m_median",
        "diff_a_m_std",
        "diff_a_m_min",
        "diff_a_m_max",
        "diff_a_m_iqr",
        "diff_dist_centroid_mean",
        "diff_dist_centroid_median",
        "diff_dist_centroid_std",
        "diff_dist_centroid_min",
        "diff_dist_centroid_max",
        "diff_dist_centroid_iqr",
        "diff_local_density_mean",
        "diff_local_density_median",
        "diff_local_density_std",
        "diff_local_density_min",
        "diff_local_density_max",
        "diff_local_density_iqr",
        "diff_avg_knn_mean",
        "diff_avg_knn_median",
        "diff_avg_knn_std",
        "diff_avg_knn_min",
        "diff_avg_knn_max",
        "diff_avg_knn_iqr",
        "diff_tot_disp",
        "diff_top_path_eff",
    ]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = axes.flatten()  # Flatten in case of multiple rows

    for i, t in zip(range(F), titles):
        ax = axes[i]
        ax.hist(data[:, i], bins=30, color='skyblue', edgecolor='black')
        ax.set_title(t)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    # Turn off any unused subplots
    for j in range(F, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_predictions(predictions: np.ndarray, names: list, filename: str = "predictions.png"):

    if not isinstance(predictions, np.ndarray) or predictions.dtype != bool:
        raise ValueError("Input must be a boolean NumPy array.")

    N = predictions.shape[0]
    M = predictions.shape[1]
    true_counts = np.sum(predictions, axis=1)  # Shape: (N,)

    if names is not None:
        if len(names) != N:
            raise ValueError(f"`names` must have length {N}, got {len(names)}.")
        x_labels = names
    else:
        x_labels = [str(i) for i in range(N)]

    x_positions = np.arange(N)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(x_positions, true_counts, color='skyblue')
    plt.xticks(x_positions, x_labels, rotation=45, ha='right')
    plt.xlabel('Object')
    plt.ylabel('Count of True Predictions')
    plt.title(f'True Prediction Counts per Object (Tot obj = {M})')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_boolean_matrices(
    main_matrix: np.ndarray,
    main_labels: list,
    secondary_matrix: np.ndarray,
    secondary_labels: list,
    output_path: str,
    figsize=(10, 6),
    title="Anomalies"
):
    """
    Save a plot of two stacked boolean matrices:
    - Top: main_matrix with main_labels on y-axis
    - Bottom: secondary_matrix with secondary_labels on y-axis
    - True = green, False = red

    Parameters
    ----------
    main_matrix : np.ndarray
        Boolean array of shape (N, T) to plot on top.

    main_labels : list
        List of N labels for the main_matrix rows.

    secondary_matrix : np.ndarray
        Boolean array of shape (M, T) to plot below.

    secondary_labels : list
        List of M labels for the secondary_matrix rows.

    output_path : str
        File path where the plot image will be saved.
    """
    assert main_matrix.dtype == bool and secondary_matrix.dtype == bool
    assert main_matrix.shape[0] == len(main_labels)
    assert secondary_matrix.shape[0] == len(secondary_labels)
    assert main_matrix.shape[1] == secondary_matrix.shape[1]  # same time axis

    fig, ax = plt.subplots(figsize=figsize)

    # Combine matrices and labels
    full_matrix = np.vstack([main_matrix, secondary_matrix])
    full_labels = main_labels + secondary_labels

    # Create colormap: False = Green, True = Red
    cmap = mcolors.ListedColormap(['green', 'red'])

    # Plot the matrix
    ax.imshow(full_matrix, aspect='auto', cmap=cmap, interpolation='nearest')

    # Add y-axis labels
    ax.set_yticks(np.arange(len(full_labels)))
    ax.set_yticklabels(full_labels)

    # Optional: label x-axis as time steps
    ax.set_xticks(np.arange(full_matrix.shape[1]))
    ax.set_xticklabels(np.arange(full_matrix.shape[1]) + 1, rotation=90)
    ax.set_xlabel("Entity")
    ax.set_title(title)

    # Add separator between main and secondary matrices
    ax.axhline(y=len(main_labels) - 0.5, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

# ========================================= END PLOTS ===============================================


def apply_robust_scaling(data: np.ndarray) -> np.ndarray:
    assert len(data.shape) == 2
    scaler = RobustScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data


def apply_standard_scaling(data: np.ndarray) -> np.ndarray:
    assert len(data.shape) == 2
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data


def apply_minmax_scaling(data: np.ndarray) -> np.ndarray:
    assert len(data.shape) == 2
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data


# PCA only when data is not affected by anomalies (or very few)
def apply_pca(
        data: np.ndarray,
        contamination: float = 0.05,
        n_components: float = 0.95,
        svd_solver: str = 'full'
) -> tuple[np.ndarray, np.ndarray]:
    assert len(data.shape) == 2
    pca = PCA(n_components=n_components, svd_solver=svd_solver)
    reduced_data = pca.fit_transform(data)
    if reduced_data.ndim == 1:  # Fix shape if PCA returns 1D
        reduced_data = reduced_data.reshape(-1, 1)

    inverse_matrix = pca.inverse_transform(reduced_data)
    errors = np.linalg.norm(data - inverse_matrix, axis=1)
    err_threshold = np.percentile(errors, 100 * (1 - contamination))
    anomalies = errors > err_threshold
    print(f"reduced to {reduced_data.shape[1]} components")
    return reduced_data, anomalies


def apply_zscore(data: np.ndarray) -> np.ndarray:
    assert len(data.shape) == 2
    z_scores = np.abs(zscore(data, axis=0))
    anomalies = np.any(z_scores > 3, axis=1)
    return anomalies


def apply_robust_zscore(data: np.ndarray) -> np.ndarray:
    assert len(data.shape) == 2
    median = np.median(data, axis=0)
    mad = np.median(np.abs(data - median), axis=0)
    mad = np.where(mad == 0, np.finfo(float).eps, mad)  # Prevent division by zero
    robust_zscore = 0.6745 * (data - median) / mad
    anomalies = np.any(np.abs(robust_zscore) > 3.5, axis=1)     # threshold 3.5 is empirically good balance
    return anomalies


def apply_iqr(data: np.ndarray) -> np.ndarray:
    assert len(data.shape) == 2
    q1, q3 = np.percentile(data, [25, 75], axis=0)
    iqr = q3 - q1
    anomalies = np.any((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr), axis=1)
    return anomalies


def apply_mahalanobis(data: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    assert len(data.shape) == 2     # ( N entities, F features)
    feat_cov = np.atleast_2d(np.cov(data.T))    # ensure 2d array, fixes collapse to shape () if num_feature is 1
    inv_cov = np.linalg.pinv(feat_cov)
    mean = np.mean(data, axis=0)
    mahal_dist = [mahalanobis(x, mean, inv_cov) for x in data]
    mahal_threshold = np.percentile(mahal_dist, (1 - contamination) * 100)
    anomalies = np.array(mahal_dist) > mahal_threshold
    return anomalies


def apply_isolation_forest(data: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    assert len(data.shape) == 2
    if_detector = IsolationForest(contamination=contamination, random_state=42)
    if_anomalies = if_detector.fit_predict(data)
    anomalies = (if_anomalies == -1)
    return anomalies


def apply_local_outlier_factor(data: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    assert len(data.shape) == 2
    lof_detector = LocalOutlierFactor(contamination=contamination, n_neighbors=min(20, data.shape[0]-1))
    lof_anomalies = lof_detector.fit_predict(data)
    anomalies = (lof_anomalies == -1)
    return anomalies


def apply_elliptic_envelope(data: np.ndarray, contamination: float = 0.05) -> np.ndarray:
    assert len(data.shape) == 2
    elenv_detector = EllipticEnvelope(contamination=contamination, random_state=42)
    elenv_anomalies = elenv_detector.fit_predict(data)
    anomalies = (elenv_anomalies == -1)
    return anomalies


def apply_dbscan(data: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    assert len(data.shape) == 2
    dbscan_detector = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_anomalies = dbscan_detector.fit_predict(data)
    anomalies = (dbscan_anomalies == -1)
    return anomalies


def apply_gaussian_mixture(data: np.ndarray, contamination: float = 0.05, n_components: int = 3) -> np.ndarray:
    assert len(data.shape) == 2
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(data)
    log_probs = gmm.score_samples(data)
    threshold = np.percentile(log_probs, contamination * 100)  # Bottom 5% = anomalies
    anomalies = log_probs < threshold
    return anomalies


def get_methods_predictions(data: np.ndarray, frame_id: int, contamination: float = 0.05) -> dict:
    # data shape: (N_objects, M_metrics)

    save_feature_distributions(data, f"TMP_HEALTH/features_{frame_id}.png")  # TODO:REMOVE

    predictions = {}

    data = apply_minmax_scaling(data)

    # ========= UNIVARIATE METHODS ================

    """
    # Z-score
    zscore_anomalies = apply_zscore(data)
    predictions["zscore"] = {
        "pred": zscore_anomalies,
        "weight": 1.0,
    }

    # IQR
    iqr_anomalies = apply_iqr(data)
    predictions["iqr"] = {
        "pred": iqr_anomalies,
        "weight": 0.7,
    }

    # Robust Z-score
    robust_zscore_anomalies = apply_robust_zscore(data)
    predictions["robust_zscore"] = {
        "pred": robust_zscore_anomalies,
        "weight": 0.7,
    }
    """

    # ========= MULTIVARIATE METHODS  ================

    # scale the data
    # data = apply_robust_scaling(data)

    save_feature_distributions(data, f"TMP_HEALTH/features_{frame_id}_scaled.png")  # TODO:REMOVE

    # Isolation Forest
    if_anomalies = apply_isolation_forest(data, contamination)
    predictions["isolation_forest"] = {
        "pred": if_anomalies,
        "weight": 1.2,
    }

    # LocalOutlierFactor
    lof_anomalies = apply_local_outlier_factor(data, contamination)
    predictions["local_outlier_factor"] = {
        "pred": lof_anomalies,
        "weight": 1.2,
    }

    """

    # DBSCAN
    dbscan_anomalies = apply_dbscan(data, eps=0.5, min_samples=5)
    predictions["dbscan"] = {
        "pred": dbscan_anomalies,
        "weight": 1.2,
    }

    # GaussianMixtureModels
    gmm_anomalies = apply_gaussian_mixture(data, contamination, n_components=3)
    predictions["gmm"] = {
        "pred": gmm_anomalies,
        "weight": 1.0,
    }

    # EllipticEnvelope (more samples than features required)
    if data.shape[0] > data.shape[1]:
        elenv_anomalies = apply_elliptic_envelope(data, contamination)
        predictions["elliptic_envelope"] = {
            "pred": elenv_anomalies,
            "weight": 1.0,
    }
    """
    
    return predictions


def get_ensemble_prediction(predictions: dict, frame_id: int, majority_vote_threshold: float):

    # Combine predictions from different methods and apply majority voting
    results = [predictions[method]["pred"] for method in predictions.keys()]
    results = np.stack(results, axis=0)  # (M methods, N anomaly predictions)

    # extract methods weighting factors
    weights = [predictions[method]["weight"] for method in predictions.keys()]  # (M methods,)

    # create ensemble anomaly score for each entity as weighted avg of methods predictions
    ensemble_votes = np.average(results.astype(float), axis=0, weights=weights)  # (N anomaly predictions)

    # if avg True weight is more than 'majority_vote_threshold' (float in [0,1)), then that entity is an anomaly
    anomalies = ensemble_votes > majority_vote_threshold

    save_predictions(results, list(predictions.keys()), f"TMP_HEALTH/features_{frame_id}_zpredict.png")  # TODO:REMOVE
    plot_boolean_matrices(results, list(predictions.keys()), anomalies[np.newaxis, :], ["ensemble"], f"TMP_HEALTH/features_{frame_id}_zmatrix.png")  # TODO:REMOVE

    return anomalies


def detect_anomalies_statistical(
        data: np.ndarray,
        frame_id: int,
        majority_vote_threshold: float,
        contamination: float = 0.05
) -> tuple[list[bool], dict]:

    predictions = get_methods_predictions(data, frame_id, contamination)
    anomalies = get_ensemble_prediction(predictions, frame_id, majority_vote_threshold)

    predictions["ensemble"] = anomalies

    return anomalies.tolist(), predictions
