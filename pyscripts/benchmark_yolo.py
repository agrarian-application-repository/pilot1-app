from ultralytics import YOLO
from src.configs.utils import read_yaml_config
import numpy as np
from time import time

if __name__ == "__main__":
    detection_args = read_yaml_config("configs/danger_detection/detector.yaml")
    detection_model_checkpoint = detection_args.pop("model_checkpoint")
    model = YOLO(detection_model_checkpoint, task="detect")
    model.export(format='engine', half=True)
    n_inters = 500
    h = 720
    w = 1280

    frames = [np.random.randint(0, 256, (h, w,3), dtype=np.uint8) for _ in range(n_inters)]
    
    start = time()
    for frame in frames:
        detection_results = model.predict(source=frame, verbose=True, **detection_args)

    print(f"{((time()-start)/n_inters*1000)} ms at {h}x{w}")


# 42.68937158584595 720,1280,3 > (1, 3, 736, 1280)
# 33.860065937042236 ms at 1280x720 > (1, 3, 736, 416)
42.68937158584595