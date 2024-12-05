import os
from argparse import ArgumentParser, Namespace

import cv2
import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def parse_cli_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def save_yolo_format(output_dir, frame_idx, boxes, labels, classes_idx, height, width):
    """
    Save bounding boxes in YOLO format.
    Each line contains: <class_index> <x_center> <y_center> <width> <height>
    All coordinates are normalized (0 to 1).
    """
    annotation_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.txt")

    with open(annotation_path, "w") as f:
        for box, label in zip(boxes, labels):
            # YOLO format requires normalized coordinates
            x_min, y_min, x_max, y_max = box
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            box_width = (x_max - x_min) / width
            box_height = (y_max - y_min) / height

            class_index = classes_idx[label]

            # Write annotation in YOLO format
            f.write(f"{class_index} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")


def main():

    args = parse_cli_args()

    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    classes_idx = {
        "a sheep": 0,
        "a goat": 1,
    }

    text_prompt = "a sheep. a goat"

    # Prepare output directories
    frames_dir = os.path.join(args.output_dir, "images")
    annotations_dir = os.path.join(args.output_dir, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}...")

        # Compare pixel values
        # print("pixel value at (0, 0):", frame[0, 0]) -> BGR
        # cv2.imwrite("./before.jpg", frame)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print("pixel value at (0, 0):", rgb_frame[0, 0]) -> RGB
        # cv2.imwrite("./after.jpg", rgb_frame)

        image = Image.fromarray(rgb_frame)

        # Prepare inputs for the model
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.4,
            text_threshold=0.3,
            target_sizes=[image.size[::-1]]
        )

        # Extract bounding boxes and save annotations
        results = results[0]
        print(results)

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        # Save frame as an image
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)

        # Save annotations in YOLO format
        height = frame.shape[0]
        width = frame.shape[1]
        save_yolo_format(annotations_dir, frame_count, boxes, labels, classes_idx, height, width)

    # Release resources
    cap.release()
    print(f"Processing complete. Frames and annotations saved in {args.output_dir}")


if __name__ == "__main__":
    main()
