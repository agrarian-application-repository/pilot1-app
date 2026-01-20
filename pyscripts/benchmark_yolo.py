from ultralytics import YOLO
from src.configs.utils import read_yaml_config
import numpy as np
from time import time

if __name__ == "__main__":
    # model = YOLO("yolo11m.pt", task="detect")
    model = YOLO("yolo11x.pt", task="detect")
    n_inters = 1000
    h = 720
    w = 1280

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
        "classes": None,
        "half": False,
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

    frames = [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(n_inters)]
    
    start = time()
    for frame in frames:
        detection_results = model.predict(source=frame, verbose=True, imgsz=1280, **yolo_args)

    print(f"{((time()-start)/n_inters*1000)} ms at {h}x{w}")


# 42.68937158584595 720,1280,3 > (1, 3, 736, 1280)
# 33.860065937042236 ms at 1280x720 > (1, 3, 736, 416)
# 42.68937158584595