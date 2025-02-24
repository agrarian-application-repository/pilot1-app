def send_alert(alerts_file, frame_id: int, num_anomalies: int):
    # Write alert to file
    alerts_file.write(
        f"Alert: "
        f"Frame {frame_id} - "
        f"{num_anomalies} instances of anomalous behaviour detected."
        f"\n"
    )
