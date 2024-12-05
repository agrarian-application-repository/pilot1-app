import cv2
from pathlib import Path
from argparse import ArgumentParser


def reduce_frame_rate(input_path: Path, output_path: Path, frames_skip: int):
    """
    Reduces the frame rate of a video by keeping one frame out of every 'frames_skip' frames.

    :param input_path: Path to the input video file.
    :param output_path: Path to save the output video file.
    :param frames_skip: Number of frames to skip. Default is 6.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format

    # Set up the video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Write one out of every 'frames_skip' frames
        if frame_count % frames_skip == 0:
            out.write(frame)

        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {output_path}")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_video_path", type=str, required=True)
    parser.add_argument("--output_video_path", type=str, required=True)
    parser.add_argument("--frames_skip", type=int, default=30)
    args = parser.parse_args()

    reduce_frame_rate(
        input_path=Path(args.input_video_path),
        output_path=Path(args.output_video_path),
        frames_skip=args.frames_skip,
    )


if __name__ == "__main__":
    main()

