import cv2
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm


def reduce_frame_rate(input_path: Path, output_dir: Path, output_name: str, sample_every_n_seconds: int):
    """
    Reduces the frame rate of a video by keeping one frame out of every 'sample_every_n_seconds' seconds.

    :param input_path: Path to the input video file.
    :param output_dir: Path to the directory where to save the output video file and images.
    :param output_name: The name of the processed output video file and images dir.
    :param sample_every_n_seconds: interval in seconds for sampling 1 frame (minimum is equal to fps of the video).
    """

    # Open the video file
    cap = cv2.VideoCapture(input_path)

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format

    # set the path to the output video
    output_video_path = output_dir / (output_name+".mp4")

    # set the path to the output images folder
    output_images_path = output_dir / output_name
    output_images_path.mkdir(parents=True, exist_ok=True)

    # Set up the video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    frame_keeper_multiplier = max(1, int(sample_every_n_seconds * fps))

    pbar = tqdm(range(total_frames), desc="Processing video")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Write one out of every 'sample_every_n_seconds' seconds
        if frame_count % frame_keeper_multiplier == 0:
            out.write(frame)
            img_path = output_images_path / f"{frame_count}.png"
            cv2.imwrite(img_path, frame)

        frame_count += 1
        pbar.update(1)

    # Release resources
    pbar.close()
    cap.release()
    out.release()
    print(f"Processed video and images saved to '{output_dir}'")


def main():
    parser = ArgumentParser()
    parser.add_argument("-in", "--input_video_path", type=str, required=True)
    parser.add_argument("-outd", "--output_dir", type=str, required=True)
    parser.add_argument("-outn", "--output_name", type=str, required=True)
    parser.add_argument("-s", "--sample_every_n_seconds", type=float, required=True)
    args = parser.parse_args()

    input_video_path = Path(args.input_video_path)
    assert input_video_path.is_file() and input_video_path.suffix.lower() == ".mp4", \
        f"Error: input_video_path is not a valid path to a '.mp4' file as expected. Got {input_video_path}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reduce_frame_rate(
        input_path=input_video_path,
        output_dir=output_dir,
        output_name=args.output_name,
        sample_every_n_seconds=args.sample_every_n_seconds,
    )


if __name__ == "__main__":
    main()

