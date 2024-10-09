from ultralytics import YOLO


def get_YOLOv11():
    model = YOLO()
    return model


def get_pretrained_YOLOv11(checkpoint: str):
    assert checkpoint in ["yolov11n.pt", "yolov11s.pt", "yolov11m.pt", "yolov11l.pt", "yolov11v11x.pt"]
    model = YOLO(checkpoint)
    return model
