def send_alert(alerts_file, frame_id: int, danger_type: str = "Generic"):
    # Write alert to file
    alerts_file.write(f"Alert: Frame {frame_id} - Animal(s) near or in dangerous area.  Danger type: {danger_type}.\n")
