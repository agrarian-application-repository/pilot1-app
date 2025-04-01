import pytest
from src.configs.drone import check_drone_args

valid_config = {
    "true_focal_len_mm": 12.29,
    "sensor_width_mm": 17.35,
    "sensor_height_mm": 13.00,
    "sensor_width_pixels": 5280,
    "sensor_height_pixels": 3956,
}


# Test valid configuration
def test_valid_config():
    assert check_drone_args(valid_config) == valid_config


@pytest.mark.parametrize("invalid_config, expected_error", [
    ({**valid_config, "extra_param": 42}, "Config must contain all and only these keys"),  # Extra key
    ({k: v for k, v in valid_config.items() if k != "sensor_height_pixels"}, "Config must contain all and only these keys"),  # Missing key
    ({**valid_config, "true_focal_len_mm": "12.29"}, "'true_focal_len_mm' must be a positive float"),  # Wrong type
    ({**valid_config, "true_focal_len_mm": 0.0}, "'true_focal_len_mm' must be a positive float"),  # Non positive
    ({**valid_config, "true_focal_len_mm": -1.0}, "'true_focal_len_mm' must be a positive float"),  # Non positive
    ({**valid_config, "sensor_width_mm": "2.0"}, "'sensor_width_mm' must be a positive float"),  # Wrong type
    ({**valid_config, "sensor_width_mm": 0.0}, "'sensor_width_mm' must be a positive float"),  # Non positive
    ({**valid_config, "sensor_width_mm": -1.0}, "'sensor_width_mm' must be a positive float"),  # Non positive
    ({**valid_config, "sensor_height_mm": "2.0"}, "'sensor_height_mm' must be a positive float"),  # Wrong type
    ({**valid_config, "sensor_height_mm": 0.0}, "'sensor_height_mm' must be a positive float"),  # Non positive
    ({**valid_config, "sensor_height_mm": -1.0}, "'sensor_height_mm' must be a positive float"),  # Non positive
    ({**valid_config, "sensor_width_pixels": "10"}, "'sensor_width_pixels' must be a positive integer"),  # Wrong type
    ({**valid_config, "sensor_width_pixels": 0}, "'sensor_width_pixels' must be a positive integer"),  # Non positive
    ({**valid_config, "sensor_width_pixels": -1}, "'sensor_width_pixels' must be a positive integer"),  # Non positive
    ({**valid_config, "sensor_height_pixels": "10"}, "'sensor_height_pixels' must be a positive integer"),  # Wrong type
    ({**valid_config, "sensor_height_pixels": 0}, "'sensor_height_pixels' must be a positive integer"),  # Non positive
    ({**valid_config, "sensor_height_pixels": -1}, "'sensor_height_pixels' must be a positive integer"),  # Non positive
    ({**valid_config, "sensor_height_pixels": -1}, "'sensor_height_pixels' must be a positive integer"),
    ({**valid_config, "sensor_height_pixels": 3400}, "Aspect ratio mismatch"),
])
def test_invalid_configs(invalid_config, expected_error):
    with pytest.raises(AssertionError, match=expected_error):
        check_drone_args(invalid_config)
