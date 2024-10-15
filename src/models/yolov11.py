from ultralytics import YOLO


def get_YOLO11():
    model = YOLO()
    return model


def get_pretrained_YOLO11(checkpoint: str):
    assert checkpoint in ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
    model = YOLO(checkpoint)
    return model
