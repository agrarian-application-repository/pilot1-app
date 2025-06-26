import pytest
import numpy as np
from src.health_monitoring.anomaly_detection.anomaly_detection import merge_previous_anomaly_status_current_detections, compute_area_fraction


# ---------------------------------------
# compute_area_fraction
# ---------------------------------------

def test_compute_area_fraction():
    radius_meters = 10.0
    meters_per_pixel = 0.05
    frame_width = 1920

    result = compute_area_fraction(radius_meters, meters_per_pixel, frame_width)
    expected = (10.0 / 0.05) / 1920  # 200 / 1920
    # 1920 = radius 1.0. ==> 200 is about 0.1

    assert np.isclose(result, expected), "Area fraction calculation is incorrect"

# ---------------------------------------
# merge_previous_anomaly_status_current_detections
# ---------------------------------------


def test_all_ids_match():
    current_ids = [1, 2, 3]
    previous_ids = [1, 2, 3]
    previous_anomaly_status = [True, False, True]
    expected = [True, False, True]
    assert merge_previous_anomaly_status_current_detections(current_ids, previous_ids, previous_anomaly_status) == expected


def test_some_ids_match_1():
    current_ids = [2, 4]
    previous_ids = [1, 2, 3]
    previous_anomaly_status = [False, True, False]
    expected = [True, False]  # 2 is True, 4 is not found => False
    assert merge_previous_anomaly_status_current_detections(current_ids, previous_ids, previous_anomaly_status) == expected


def test_some_ids_match_2():
    current_ids = [2, 4]
    previous_ids = [1, 2, 3]
    previous_anomaly_status = [False, False, True]
    expected = [False, False]  # 2 is False, 4 is not found => False
    assert merge_previous_anomaly_status_current_detections(current_ids, previous_ids, previous_anomaly_status) == expected


def test_no_ids_match():
    current_ids = [10, 11]
    previous_ids = [1, 2]
    previous_anomaly_status = [True, False]
    expected = [False, False]
    assert merge_previous_anomaly_status_current_detections(current_ids, previous_ids, previous_anomaly_status) == expected


def test_empty_previous():
    current_ids = [1, 2, 3]
    previous_ids = []
    previous_anomaly_status = []
    expected = [False, False, False]
    assert merge_previous_anomaly_status_current_detections(current_ids, previous_ids, previous_anomaly_status) == expected


def test_empty_current():
    current_ids = []
    previous_ids = [1, 2]
    previous_anomaly_status = [True, False]
    expected = []
    assert merge_previous_anomaly_status_current_detections(current_ids, previous_ids, previous_anomaly_status) == expected


def test_empty_all():
    current_ids = []
    previous_ids = []
    previous_anomaly_status = []
    expected = []
    assert merge_previous_anomaly_status_current_detections(current_ids, previous_ids, previous_anomaly_status) == expected


def test_mismatched_lengths_raises_error():
    current_ids = [1, 2]
    previous_ids = [1]
    previous_anomaly_status = [True, False]  # len mismatch
    with pytest.raises(AssertionError):
        merge_previous_anomaly_status_current_detections(current_ids, previous_ids, previous_anomaly_status)
