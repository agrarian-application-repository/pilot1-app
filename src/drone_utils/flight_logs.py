import re


def _parse_line_to_dict(line):
    # Regex to capture key-value pairs
    pattern = r'(\w+)\s*:\s*([\w\.\-/]+)'

    # Use re.findall to extract all key-value pairs
    matches = re.findall(pattern, line)

    # Convert the matches to a dictionary
    result = {key: _convert_value(value) for key, value in matches}

    return result


def _convert_value(value):
    """Convert the value to the appropriate type: int, float, or keep as string."""
    try:
        # Attempt to convert to int
        return int(value)
    except ValueError:
        try:
            # Attempt to convert to float
            return float(value)
        except ValueError:
            # Keep as string if it can't be converted
            return value


def parse_drone_flight_data(file, frame_id):
    # Read all lines from the file
    lines = file.readlines()

    # Calculate the starting line for the desired frame_id
    # Each frame has 6 lines: 1 frame number + 4 data lines + 1 space before next frame
    frame_start_line = (frame_id - 1) * 6
    # Check if the required lines are within the file's length
    if frame_start_line + 5 > len(lines):
        raise ValueError(f"Frame {frame_id} not found or in the file (frames info are supposed to be sequential).")

    # Extract the 5 lines for the desired frame
    frame_lines = lines[frame_start_line:frame_start_line + 5]

    # The useful line is the 5th line in the block (index 4)
    useful_line = frame_lines[4].strip()

    # rewind to file beginning to avoid error on next readlines()
    file.seek(0)

    return _parse_line_to_dict(useful_line)