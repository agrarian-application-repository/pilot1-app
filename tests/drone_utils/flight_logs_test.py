import pytest
from src.drone_utils.flight_logs import parse_drone_flight_data

# Sample content to test
content = """
1\n00:00:00,000 --> 00:00:00,033\n<font size="28">FrameCnt: 1, DiffTime: 33ms\n2024-10-24 10:49:35.192\n[iso: 100] [shutter: 1/573.66] [fnum: 2.8] [ev: 0] [color_md : default] [ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 35.427071] [longitude: 24.174011] [rel_alt: 27.531 abs_alt: 180.971] [gb_yaw: -35.1 gb_pitch: -89.1 gb_roll: 0.0] </font>\n,
2\n00:00:00,033 --> 00:00:00,066\n<font size="28">FrameCnt: 2, DiffTime: 33ms\n2024-10-24 10:49:35.224\n[iso: 100] [shutter: 1/573.66] [fnum: 2.8] [ev: 0] [color_md : default] [ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 35.427072] [longitude: 24.174012] [rel_alt: 27.532 abs_alt: 180.972] [gb_yaw: -35.2 gb_pitch: -89.2 gb_roll: 0.0] </font>\n,
3\n00:00:00,066 --> 00:00:00,099\n<font size="28">FrameCnt: 3, DiffTime: 33ms\n2024-10-24 10:49:35.260\n[iso: 100] [shutter: 1/573.66] [fnum: 2.8] [ev: 0] [color_md : default] [ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 35.427073] [longitude: 24.174013] [rel_alt: 27.533 abs_alt: 180.973] [gb_yaw: -35.3 gb_pitch: -89.3 gb_roll: 0.0] </font>\n,
4\n00:00:00,099 --> 00:00:00,133\n<font size="28">FrameCnt: 4, DiffTime: 34ms\n2024-10-24 10:49:35.295\n[iso: 100] [shutter: 1/573.66] [fnum: 2.8] [ev: 0] [color_md : default] [ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 35.427074] [longitude: 24.174014] [rel_alt: 27.534 abs_alt: 180.974] [gb_yaw: -35.4 gb_pitch: -90.4 gb_roll: 0.0] </font>\n,
5\n00:00:00,133 --> 00:00:00,166\n<font size="28">FrameCnt: 5, DiffTime: 33ms\n2024-10-24 10:49:35.326\n[iso: 100] [shutter: 1/579.91] [fnum: 2.8] [ev: 0] [color_md : default] [ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 35.427075] [longitude: 24.174015] [rel_alt: 27.535 abs_alt: 180.975] [gb_yaw: -35.5 gb_pitch: -90.5 gb_roll: 0.0] </font>\n,
6\n00:00:00,166 --> 00:00:00,199\n<font size="28">FrameCnt: 6, DiffTime: 33ms\n2024-10-24 10:49:35.358\n[iso: 100] [shutter: 1/579.91] [fnum: 2.8] [ev: 0] [color_md : default] [ae_meter_md: 1] [focal_len: 24.00] [dzoom_ratio: 1.00], [latitude: 35.427076] [longitude: 24.174016] [rel_alt: 27.536 abs_alt: 180.976] [gb_yaw: -35.6 gb_pitch: -90.6 gb_roll: 0.0] </font>\n,
"""

example_file = 'tests/drone_utils/flight_logs.txt'

# Test Data
expected_data = {
    1: {"iso": 100, "shutter": "1/573.66", "fnum": 2.8, "ev": 0, "color_md": "default", "ae_meter_md": 1, "focal_len": 24.0, "dzoom_ratio": 1.0, "latitude": 35.427071, "longitude": 24.174011, "rel_alt": 27.531, "abs_alt": 180.971, "gb_yaw": -35.1, "gb_pitch": -89.1, "gb_roll": 0.0},
    2: {"iso": 100, "shutter": "1/573.66", "fnum": 2.8, "ev": 0, "color_md": "default", "ae_meter_md": 1, "focal_len": 24.0, "dzoom_ratio": 1.0, "latitude": 35.427072, "longitude": 24.174012, "rel_alt": 27.532, "abs_alt": 180.972, "gb_yaw": -35.2, "gb_pitch": -89.2, "gb_roll": 0.0},
    3: {"iso": 100, "shutter": "1/573.66", "fnum": 2.8, "ev": 0, "color_md": "default", "ae_meter_md": 1, "focal_len": 24.0, "dzoom_ratio": 1.0, "latitude": 35.427073, "longitude": 24.174013, "rel_alt": 27.533, "abs_alt": 180.973, "gb_yaw": -35.3, "gb_pitch": -89.3, "gb_roll": 0.0},
    4: {"iso": 100, "shutter": "1/573.66", "fnum": 2.8, "ev": 0, "color_md": "default", "ae_meter_md": 1, "focal_len": 24.0, "dzoom_ratio": 1.0, "latitude": 35.427074, "longitude": 24.174014, "rel_alt": 27.534, "abs_alt": 180.974, "gb_yaw": -35.4, "gb_pitch": -90.4, "gb_roll": 0.0},
    5: {"iso": 100, "shutter": "1/579.91", "fnum": 2.8, "ev": 0, "color_md": "default", "ae_meter_md": 1, "focal_len": 24.0, "dzoom_ratio": 1.0, "latitude": 35.427075, "longitude": 24.174015, "rel_alt": 27.535, "abs_alt": 180.975, "gb_yaw": -35.5, "gb_pitch": -90.5, "gb_roll": 0.0},
    6: {"iso": 100, "shutter": "1/579.91", "fnum": 2.8, "ev": 0, "color_md": "default", "ae_meter_md": 1, "focal_len": 24.0, "dzoom_ratio": 1.0, "latitude": 35.427076, "longitude": 24.174016, "rel_alt": 27.536, "abs_alt": 180.976, "gb_yaw": -35.6, "gb_pitch": -90.6, "gb_roll": 0.0},
}


# Pytest function to check for the correct dictionary output
def test_parse_frame_data():

    with open(example_file) as tmp_file:
        for frame_id, expected in expected_data.items():
            # Call the function with the frame_id and content
            result = parse_drone_flight_data(tmp_file, frame_id)

            # Check that the result matches the expected output
            assert result == expected, f"Test failed for frame {frame_id}.\nGot: {result}\nExpected:{expected}"


# Pytest function to handle errors when frame_id doesn't exist
def test_parse_frame_data_error():
    # List of frame_ids that should raise an error (they don't exist)
    invalid_frame_ids = [7, 8, 9]

    with open(example_file) as tmp_file:
        for frame_id in invalid_frame_ids:
            # Check that the function raises an error when the frame_id doesn't exist
            with pytest.raises(ValueError):
                parse_drone_flight_data(tmp_file, frame_id)
