from typing import Any


def check_danger_detection_args(args: dict[str, Any]) -> dict[str, Any]:

    required_keys = {
        "safety_radius_m",
        "slope_angle_threshold",
        "alerts_cooldown_seconds",
        "geofencing_vertexes",
    }

    # Check if all required keys exist and no extra keys are present
    assert set(args.keys()) == required_keys, \
        f"Config must contain all and only these keys: {required_keys}. Got {set(args.keys())}"

    assert isinstance(args['safety_radius_m'], float) and args['safety_radius_m'] > 0, \
        f"'safety_radius_m' must be a positive float. Got {args['safety_radius_m']}"

    assert isinstance(args['slope_angle_threshold'], float) and args['slope_angle_threshold'] > 0, \
        f"'slope_angle_threshold' must be a positive float. Got {args['slope_angle_threshold']}"

    assert isinstance(args['alerts_cooldown_seconds'], int) and args['alerts_cooldown_seconds'] > 0, \
        f"'alerts_cooldown_seconds' must be a positive integer. Got {args['alerts_cooldown_seconds']}"

    assert args['geofencing_vertexes'] is None or (
            isinstance(args['geofencing_vertexes'], list) and
            len(args['geofencing_vertexes']) >= 3 and
            all(
                isinstance(coords, list) and
                len(coords) == 2 and
                isinstance(coords[0], float) and -180 <= coords[0] <= 180 and
                isinstance(coords[1], float) and -90 <= coords[1] <= 90
                for coords in args['geofencing_vertexes']
            )), \
        f"'geofencing_vertexes' must be a list of (lng,lat) coordinates. Got {args['geofencing_vertexes']}"

    return args
