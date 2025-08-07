import pytest
from src.configs.danger_detection import check_danger_detection_args

# Valid configuration for reference
VALID_CONFIG = {
    "safety_radius_m": 1.5,
    "slope_angle_threshold": 30.0,
    "geofencing_vertexes": [
        [21.200000, 34.0],
        [24.174019, 34.0],
        [24.174019, 36.0],
        [21.200000, 36.0]
    ],
    "alerts_cooldown_seconds": 5,
}


# Test valid configuration
def test_valid_config():
    assert check_danger_detection_args(VALID_CONFIG) == VALID_CONFIG


# Parameterized invalid configurations
@pytest.mark.parametrize("invalid_config, expected_error", [
    # Missing key
    ({**VALID_CONFIG, "extra_param": 42}, "Config must contain all and only these keys"),
    ({k: v for k, v in VALID_CONFIG.items() if k != "safety_radius_m"}, "Config must contain all and only these keys"),

    # Invalid types
    ({**VALID_CONFIG, "alerts_cooldown_seconds": 5.0}, "'alerts_cooldown_seconds' must be a positive integer"),
    ({**VALID_CONFIG, "safety_radius_m": "1.5"}, "'safety_radius_m' must be a positive float"),

    # Invalid values
    ({**VALID_CONFIG, "safety_radius_m": -1.0}, "'safety_radius_m' must be a positive float"),
    ({**VALID_CONFIG, "alerts_cooldown_seconds": -10}, "'alerts_cooldown_seconds' must be a positive integer"),

    # Empty list
    ({**VALID_CONFIG, "geofencing_vertexes": []}, "'geofencing_vertexes' must be a list of "),
    # Not a list
    ({**VALID_CONFIG, "geofencing_vertexes": "not_a_list"}, "'geofencing_vertexes' must be a list of "),
    # Nested list structure incorrect
    ({**VALID_CONFIG, "geofencing_vertexes": [21.2, 34.0, 22.0, 35.0]}, "'geofencing_vertexes' must be a list of "),
    # Not enough vertexes (less than 3)
    ({**VALID_CONFIG, "geofencing_vertexes": [[21.2, 34.0], [24.17]]}, "'geofencing_vertexes' must be a list of "),
    ({**VALID_CONFIG, "geofencing_vertexes": [[21.2, 34.0]]}, "'geofencing_vertexes' must be a list of "),
    # not two coords
    ({**VALID_CONFIG, "geofencing_vertexes": [[160.0, 34.0], [21.0, 30.0], [31.0]]}, "'geofencing_vertexes' must be a list of "),
    ({**VALID_CONFIG, "geofencing_vertexes": [[160.0, 34.0], [21.0, 30.0], [22.0, 31.0, 44.0]]}, "'geofencing_vertexes' must be a list of "),
    # Longitude out of bounds (> 180)
    ({**VALID_CONFIG, "geofencing_vertexes": [[190.0, 34.0], [21.0, 30.0], [22.0, 31.0]]}, "'geofencing_vertexes' must be a list of "),
    # Longitude out of bounds (< -180)
    ({**VALID_CONFIG, "geofencing_vertexes": [[47.0, 34.0], [21.0, 30.0], [-182.0, 31.0]]},
     "'geofencing_vertexes' must be a list of "),
    # Latitude out of bounds (< -90)
    ({**VALID_CONFIG, "geofencing_vertexes": [[21.2, -95.0], [22.0, 34.0], [23.0, 35.0]]}, "'geofencing_vertexes' must be a list of "),
    # Latitude out of bounds (> 90)
    ({**VALID_CONFIG, "geofencing_vertexes": [[21.2, 34.0], [22.0, 95.0], [23.0, 35.0]]}, "'geofencing_vertexes' must be a list of "),
    # Non-float values in vertexes
    ({**VALID_CONFIG, "geofencing_vertexes": [["21.2", 34.0], [22.0, 35.0], [23.0, 36.0]]}, "'geofencing_vertexes' must be a list of "),
    ({**VALID_CONFIG, "geofencing_vertexes": [[21.2, "34.0"], [22.0, 35.0], [23.0, 36.0]]},"'geofencing_vertexes' must be a list of "),
    # Mixed valid and invalid coordinates
    ({**VALID_CONFIG, "geofencing_vertexes": [[21.2, 34.0], [22.0, 35.0], ["invalid", 36.0]]}, "'geofencing_vertexes' must be a list of "),
])
def test_invalid_configs(invalid_config, expected_error):
    with pytest.raises(AssertionError, match=expected_error):
        check_danger_detection_args(invalid_config)
