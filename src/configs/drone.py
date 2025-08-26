from typing import Any


def check_drone_args(args: dict[str, Any]) -> dict[str, Any]:
    required_keys = {
        "true_focal_len_mm",
        "sensor_width_mm",
        "sensor_height_mm",
        "sensor_width_pixels",
        "sensor_height_pixels",
        "frame_width",
        "frame_height",
        "fps",
    }

    # Check if all required keys exist and no extra keys are present
    assert set(args.keys()) == required_keys, \
        f"Config must contain all and only these keys: {required_keys}. Got {set(args.keys())}"

    assert isinstance(args['true_focal_len_mm'], float) and args['true_focal_len_mm'] > 0, \
        f"'true_focal_len_mm' must be a positive float. Got {args['true_focal_len_mm']}"

    assert isinstance(args['sensor_width_mm'], float) and args['sensor_width_mm'] > 0, \
        f"'sensor_width_mm' must be a positive float. Got {args['sensor_width_mm']}"

    assert isinstance(args['sensor_height_mm'], float) and args['sensor_height_mm'] > 0, \
        f"'sensor_height_mm' must be a positive float. Got {args['sensor_height_mm']}"

    assert isinstance(args['sensor_width_pixels'], int) and args['sensor_width_pixels'] > 0, \
        f"'sensor_width_pixels' must be a positive integer. Got {args['sensor_width_pixels']}"

    assert isinstance(args['sensor_height_pixels'], int) and args['sensor_height_pixels'] > 0, \
        f"'sensor_height_pixels' must be a positive integer. Got {args['sensor_height_pixels']}"
    
    # Check aspect ratio condition
    aspect_ratio_mm = args["sensor_width_mm"] / args["sensor_height_mm"]
    aspect_ratio_pixels = args["sensor_width_pixels"] / args["sensor_height_pixels"]
    assert abs(aspect_ratio_mm - aspect_ratio_pixels) < 1e-2, \
        f"Aspect ratio mismatch: sensor_width_mm/sensor_height_mm ({aspect_ratio_mm:.6f}) \
        does not match sensor_width_pixels/sensor_height_pixels ({aspect_ratio_pixels:.6f})"
    
    assert isinstance(args['frame_width'], int) and args['frame_width'] >= 32, \
        f"'frame_width' must be a positive integer >= 32. Got {args['frame_width']}"
    
    assert isinstance(args['frame_height'], int) and args['frame_height'] >= 32, \
        f"'frame_height' must be a positive integer >= 32. Got {args['frame_height']}"
    
    assert isinstance(args['fps'], int) and args['fps'] > 0, \
        f"'fps' must be a positive integer. Got {args['fps']}"

    return args
