import io

from src.in_danger.output.alerts import send_alert


def test_send_alert_default():
    """Test that send_alert writes the expected message with the default danger type."""
    fake_file = io.StringIO()
    frame_id = 10
    send_alert(fake_file, frame_id)

    # Reset the cursor to the beginning to read its content.
    fake_file.seek(0)
    expected_output = f"Alert: Frame {frame_id} - Animal(s) near or in dangerous area.  Danger type: Generic.\n"
    assert fake_file.read() == expected_output


def test_send_alert_custom_danger():
    """Test that send_alert writes the expected message with a custom danger type."""
    fake_file = io.StringIO()
    frame_id = 20
    custom_danger = "High"
    send_alert(fake_file, frame_id, danger_type=custom_danger)

    fake_file.seek(0)
    expected_output = f"Alert: Frame {frame_id} - Animal(s) near or in dangerous area.  Danger type: {custom_danger}.\n"
    assert fake_file.read() == expected_output


def test_send_alert_multiple_calls():
    """Test that multiple calls to send_alert append to the file correctly."""
    fake_file = io.StringIO()

    # First alert with custom danger type.
    send_alert(fake_file, 1, danger_type="Low")
    # Second alert with default danger type.
    send_alert(fake_file, 2)

    fake_file.seek(0)
    expected_output = (
        "Alert: Frame 1 - Animal(s) near or in dangerous area.  Danger type: Low.\n"
        "Alert: Frame 2 - Animal(s) near or in dangerous area.  Danger type: Generic.\n"
    )
    assert fake_file.read() == expected_output
