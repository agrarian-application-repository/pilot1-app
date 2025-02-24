import cv2
import time
import numpy as np
import csv

# Define different resolutions to test
resolutions = [
    (1920, 1080),  # Full HD
    (1280, 720),   # HD
    (640, 480),    # Standard
]

# Define codecs to test
codecs = {
    # "MJPG": cv2.VideoWriter_fourcc(*"MJPG"), #slow
    "XVID": cv2.VideoWriter_fourcc(*"XVID"),
    "MP4V": cv2.VideoWriter_fourcc(*"mp4v"),
}

# Benchmark parameters
num_frames = 1000  # Number of frames to write
fps = 30          # Frames per second
output_file = "benchmark_results.csv"

# Open CSV file for writing results
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Resolution", "Codec", "Total Time (s)", "Avg Time per Frame (ms)"])

    for width, height in resolutions:
        for codec_name, fourcc in codecs.items():
            print(f"Testing {codec_name} at {width}x{height}...")

            # Create VideoWriter object
            extension = "avi" if codec_name in ["MJPG", "XVID"] else "mp4"
            out = cv2.VideoWriter(f"test_output.{extension}", fourcc, fps, (width, height))

            # Generate a dummy frame (random colors)
            frame = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

            # Start timing
            start_time = time.time()

            # Write frames
            for _ in range(num_frames):
                out.write(frame)

            # Stop timing
            total_time = time.time() - start_time
            avg_time_per_frame = (total_time / num_frames) * 1000  # Convert to ms

            # Save results
            writer.writerow([f"{width}x{height}", codec_name, round(total_time, 4), round(avg_time_per_frame, 4)])

            print(f"✔ Done: {total_time:.2f}s total, {avg_time_per_frame:.2f}ms per frame")

            # Release writer
            out.release()

print(f"\n Benchmark completed! Results saved to {output_file}.")
