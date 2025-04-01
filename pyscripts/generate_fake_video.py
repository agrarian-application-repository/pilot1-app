import cv2
import numpy as np
from pathlib import Path


def generate_fake_video(output_path, width=640, height=480, duration=10, fps=30):
    """
    Generates a fake MP4 video for testing purposes.

    :param output_path: Path where the video file will be saved.
    :param width: Width of the video frame.
    :param height: Height of the video frame.
    :param duration: Duration of the video in seconds.
    :param fps: Frames per second.
    """
    output_path = Path(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    num_frames = duration * fps

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create a moving rectangle for simple animation
        rect_x = (i * 10) % width
        cv2.rectangle(frame, (rect_x, 100), (rect_x + 50, 200), (0, 255, 0), -1)

        out.write(frame)

    out.release()
    print(f"Fake video saved at: {output_path}")


if __name__ == "__main__":
    generate_fake_video("test_video_1.mp4")
    generate_fake_video("test_video_2.mp4")
    generate_fake_video("test_video_3.mp4")
