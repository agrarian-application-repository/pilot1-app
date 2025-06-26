from src.health_monitoring.anomaly_detection.statistical_methods import get_ensemble_prediction

import numpy as np
import pytest


@pytest.fixture
def sample_predictions():
    predictions = {
        "a": {"pred": [True, False, True],      "weight": 0.7},
        "b": {"pred": [True, False, False],     "weight": 0.7},
        "c": {"pred": [True, True, True],       "weight": 1.2},
        "d": {"pred": [False, False, False],    "weight": 1.2},
        "e": {"pred": [False, True, True],      "weight": 1.2}
    }
    return predictions


def test_ensemble_prediction_shape(sample_predictions):
    result = get_ensemble_prediction(sample_predictions, frame_id=0, majority_vote_threshold=0.5)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3,)  # Three entities
    assert result.dtype == bool


def test_ensemble_prediction_values(sample_predictions):
    result = get_ensemble_prediction(sample_predictions, frame_id=0, majority_vote_threshold=0.5)

    # Manual ensemble vote computation:
    # Entity 0: weighted avg = (0.7+0.7+1.2+0+0)/5 = 2.6/5 = 0.52 → True
    # Entity 1: (0+0+1.2+0+1.2)/5 = 2.4/5 = 0.48 → False
    # Entity 2: (0.7+0+1.2+0+1.2)/5 = 3.1/5 = 0.62 → True
    expected = np.array([True, False, True])
    assert np.array_equal(result, expected)


def test_ensemble_prediction_values_high_th(sample_predictions):
    result = get_ensemble_prediction(sample_predictions, frame_id=0, majority_vote_threshold=0.6)

    # Manual ensemble vote computation:
    # Entity 0: weighted avg = (0.7+0.7+1.2+0+0)/5 = 2.6/5 = 0.52 → True
    # Entity 1: (0+0+1.2+0+1.2)/5 = 2.4/5 = 0.48 → False
    # Entity 2: (0.7+0+1.2+0+1.2)/5 = 3.1/5 = 0.62 → True
    expected = np.array([False, False, True])
    assert np.array_equal(result, expected)


def test_ensemble_prediction_values_low_th(sample_predictions):
    result = get_ensemble_prediction(sample_predictions, frame_id=0, majority_vote_threshold=0.45)

    # Manual ensemble vote computation:
    # Entity 0: weighted avg = (0.7+0.7+1.2+0+0)/5 = 2.6/5 = 0.52 → True
    # Entity 1: (0+0+1.2+0+1.2)/5 = 2.4/5 = 0.48 → False
    # Entity 2: (0.7+0+1.2+0+1.2)/5 = 3.1/5 = 0.62 → True
    expected = np.array([True, True, True])
    assert np.array_equal(result, expected)


def test_ensemble_prediction_all_agree():
    predictions = {
        "a": {"pred": [True, True, True], "weight": 1.0},
        "b": {"pred": [True, True, True], "weight": 1.0}
    }
    result = get_ensemble_prediction(predictions, frame_id=1, majority_vote_threshold=0.5)
    expected = np.array([True, True, True])
    assert np.array_equal(result, expected)


def test_ensemble_prediction_all_disagree():
    predictions = {
        "a": {"pred": [True, True, True], "weight": 1.0},
        "b": {"pred": [False, False, False], "weight": 1.0}
    }
    result = get_ensemble_prediction(predictions, frame_id=1, majority_vote_threshold=0.5)
    expected = np.array([False, False, False])  # 50% vote → not > 0.5
    assert np.array_equal(result, expected)


def test_ensemble_prediction_with_zero_weights():
    predictions = {
        "a": {"pred": [True, False, True], "weight": 0.0},
        "b": {"pred": [False, True, False], "weight": 0.0},
        "c": {"pred": [True, True, True], "weight": 1.0}
    }
    result = get_ensemble_prediction(predictions, frame_id=2, majority_vote_threshold=0.5)
    expected = np.array([True, True, True])  # Only 'c' counts
    assert np.array_equal(result, expected)
