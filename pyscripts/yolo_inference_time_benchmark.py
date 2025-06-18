import argparse
import time
import cv2
import csv
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark YOLOv11 inference time over images of varying resolutions'
    )
    parser.add_argument(
        '--image', type=str, required=False,
        default="/archive/group/ai/datasets/AGRARIAN/manual_annotations/test_splitting/images/00d41c29-DJI_20250306132455_0001_D_9492.png",
        help='Path to the input image file'
    )
    parser.add_argument(                           # train_imgsz: 640, 480 , 320
        '--imgsz', type=int, nargs='+', required=False, default=[1920, 1440, 960],
        help='List of image widths to test'
    ) # 1920->1088, 1440->832, 960->544
    parser.add_argument(
        '--runs', type=int, default=200,
        help='Number of inference runs to average'
    )
    parser.add_argument(
        '--warmup', type=int, default=20,
        help='Number of warmup runs before timing'
    )
    parser.add_argument(
        '--output', type=str, default='benchmark_results.csv',
        help='Path to save the CSV results file'
    )
    args = parser.parse_args()

    return args


def load_model(model_path):
    # Load YOLOv11 model using updated ultralytics API
    model = YOLO(model_path, task="detect")
    model.to("cuda")
    return model


def benchmark(model, img, warmup, runs, yolo_args):
    # Warmup
    for _ in range(warmup):
        _ = model.predict(source=img, **yolo_args)
    # Timed runs
    start = time.perf_counter()
    for _ in range(runs):
        _ = model.predict(source=img, **yolo_args)
    total = time.perf_counter() - start
    return total / runs * 1000


def onnx_predict(img, w, h, session, input_name):
    resized = cv2.resize(img, (w, h))
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    outputs = session.run(None, {input_name: img})  # output is a list
    predictions = outputs[0]  # shape: (1, num_detections, 6)

    boxes = []
    conf_thresh = 0.3
    for det in predictions[0]:
        x1, y1, x2, y2, conf, cls = det
        if conf > conf_thresh:
            boxes.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "conf": float(conf),
                "class": int(cls)
            })


def benchmark_onnx(warmup, runs, img, w, h, session, input_name):
    # Warmup
    for _ in range(warmup):
        onnx_predict(img, w, h, session, input_name)
    # Timed runs
    start = time.perf_counter()
    for _ in range(runs):
        onnx_predict(img, w, h, session, input_name)
    total = time.perf_counter() - start
    return total / runs * 1000


def save_results_csv(headers, results, output_path):
    # Write results list of tuples to CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in results:
            writer.writerow(row)


def main():
    args = parse_args()

    # Load image (1920x1080)
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")

    headers = ['Model', 'Imgsz', 'Half', 'Avg Time (ms)']
    ckpts = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
    use_half = [False, True]
    yolo_args = {
        "conf": 0.3,
        "iou": 0.3,
        "device": "cuda",
        "batch": 1,
        "max_det": 500,
        "vid_stride": 1,
        "stream_buffer": False,
        "visualize": False,
        "augment": False,
        "agnostic_nms": False,
        "classes": [18],
        "retina_masks": False,
        "embed": None,
        "project": None,
        "name": None,
        "show": False,
        "save": False,
        "save_frames": False,
        "save_txt": False,
        "save_conf": False,
        "save_crop": False,
        "show_labels": False,
        "show_conf": False,
        "show_boxes": False,
        "line_width": None,
    }

    results = []
    for model_ckpt in ckpts:
        model = load_model(model_ckpt)
        for imgsz in args.imgsz:
            yolo_args["imgsz"] = imgsz
            for half in use_half:
                yolo_args["half"] = half
                avg_time = benchmark(model, img, args.warmup, args.runs, yolo_args)
                results.append((model_ckpt, imgsz, half, round(avg_time, 2)))
    """
    for model_ckpt in ckpts:
        for (w,h) in zip(args.imgsz, [1088, 832,544]):
            model = YOLO(model_ckpt, task="detect")
            model.export(
                format="onnx",
                imgsz=(h,w),
                half=False,
                nms=True,
                device=0,
                conf=0.3,
                iou=0.3,
                agnostic_nms=False,
            )

            model_name = model_ckpt.split(".")[0]+".onnx"
            session = ort.InferenceSession(model_name)  # or your chosen model
            input_name = session.get_inputs()[0].name

            avg_time = benchmark_onnx(args.warmup, args.runs, img, w, h, session, input_name)
            print("AVG TIME:", avg_time)
            results.append((model_ckpt, f"{w}x{h}", False, round(avg_time, 2)))
    """

    # Save to CSV
    save_results_csv(headers, results, args.output)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()
