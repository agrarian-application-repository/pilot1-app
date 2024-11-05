from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO, solutions


def heatmap_in_video(yolo_config: dict[str:Any]) -> None:
    cap = cv2.VideoCapture(str(yolo_config["source"]))
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    # Video writer
    video_writer = cv2.VideoWriter(
        "heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    # Init heatmap
    heatmap = solutions.Heatmap(
        model=yolo_config.pop("model"),
        classes=yolo_config["classes"],
        show=False,
        show_in=False,
        show_out=False,
        colormap=cv2.COLORMAP_INFERNO,
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print(
                "Video frame is empty or video processing has been successfully completed."
            )
            break
        frame = heatmap.generate_heatmap(frame)
        video_writer.write(frame)

    cap.release()
    video_writer.release()
