from argparse import ArgumentParser, Namespace
from pathlib import Path
import shutil
import numpy as np
import cv2


def parse_cli() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    return args


def preprocess_cli_args(args: Namespace) -> tuple[Path, Path]:
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    assert in_dir.exists(), f"{in_dir} does not exists"
    assert not out_dir.exists(), f"{out_dir} already exists"
    out_dir.mkdir(exist_ok=False)
    return in_dir, out_dir


def create_out_dir_structure(in_dir: Path, out_dir: Path):
    (out_dir/"images").mkdir(exist_ok=False)
    (out_dir/"labels").mkdir(exist_ok=False)
    shutil.copy(in_dir/"classes.txt", out_dir/"classes.txt")
    shutil.copy(in_dir/"notes.json", out_dir/"notes.json")


def read_yolo_annotation(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(file_path)
    print(data)

    # handle case no bounding boxes
    if data.size == 0:
        classes = np.empty((0, 1))
        bboxes = np.empty((0, 4))
        return classes, bboxes

    # handle case with one annotation
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)

    classes = data[:, 0].astype(int).copy()
    bboxes = data[:, 1:].copy()

    return classes, bboxes


def xywhn2xyxy(x: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Convert normalized bounding box coordinates to pixel coordinates.

    Args:
        x (np.ndarray): The bounding box coordinates.
        w (int): Width of the image.
        h (int): Height of the image.

    Returns:
        y (np.ndarray): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
            x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
    """

    # handle case no bounding boxes
    if x.size == 0:
        return np.empty((0, 4))

    assert x.shape[1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)  # faster than clone/copy
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2)  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2)  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2)  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2)  # bottom right y
    return y


def xyxy2xywhn(x: np.ndarray, w: int, h: int):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
    width and height are normalized to image dimensions.

    Args:
        x (np.ndarray): The input bounding box coordinates in (x1, y1, x2, y2) format.
        w (int): The width of the image.
        h (int): The height of the image.

    Returns:
        y (np.ndarray): The bounding box coordinates in (x, y, width, height, normalized) format
    """

    # handle case no bounding boxes
    if x.size == 0:
        return np.empty((0, 4))

    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = np.empty_like(x)  # faster than clone/copy
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2) / w  # x center
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2) / h  # y center
    y[..., 2] = (x[..., 2] - x[..., 0]) / w  # width
    y[..., 3] = (x[..., 3] - x[..., 1]) / h  # height
    return y


# TODO: FIX (single split & split > img_dim)
def get_split_ranges(img_dim: int, num_splits: int, split_dim: int):
    step = (img_dim - split_dim) / (num_splits - 1)
    return [(int(i * step), int(i * step + split_dim - 1)) for i in range(num_splits)]


def save_image_crops(img: np.ndarray, img_path: Path, target_img_dir: Path, w_ranges, h_ranges):
    i = 0
    for h_range in h_ranges:
        for w_range in w_ranges:
            y1, y2 = h_range
            x1, x2 = w_range
            crop = img[y1:y2 + 1, x1:x2 + 1]

            img_name = img_path.stem
            img_suffix = img_path.suffix
            crop_path = target_img_dir/f"{img_name}_{i}{img_suffix}"
            cv2.imwrite(str(crop_path), crop)

            i += 1


def save_labels_crops(classes: np.ndarray, xyxy_bboxes: np.ndarray, label_path: Path, target_labels: Path, w_ranges, h_ranges):
    i = 0
    for h_range in h_ranges:
        for w_range in w_ranges:
            y1, y2 = h_range
            x1, x2 = w_range

            crop_classes = []
            crop_bboxes = []

            for class_label, bbox in zip(classes, xyxy_bboxes):

                # Calculate intersection
                inter_x1 = max(bbox[0], x1)
                inter_y1 = max(bbox[1], y1)
                inter_x2 = min(bbox[2], x2)
                inter_y2 = min(bbox[3], y2)
                if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                    crop_classes.append([class_label])
                    crop_bboxes.append([inter_x1 - x1, inter_y1 - y1, inter_x2 - x1, inter_y2 - y1])

            crop_classes = np.array(crop_classes)
            crop_bboxes = np.array(crop_bboxes)

            if crop_classes.size != 0:
                crop_normalized_bboxes = xyxy2xywhn(crop_bboxes, w=int(x2-x1+1), h=int(y2-y1+1))
                crop_labels = np.concatenate((crop_classes, crop_normalized_bboxes), axis=1)
            else:
                crop_labels = np.empty((0, 5))

            label_name = label_path.stem
            label_suffix = label_path.suffix
            label_crop_path = target_labels / f"{label_name}_{i}{label_suffix}"
            np.savetxt(label_crop_path, crop_labels, fmt="%d %.16f %.16f %.16f %.16f")

            i += 1


def main():

    args = parse_cli()
    in_dir, out_dir = preprocess_cli_args(args)
    create_out_dir_structure(in_dir, out_dir)

    original_images = in_dir/"images"
    original_labels = in_dir/"labels"

    target_images = out_dir/"images"
    target_labels = out_dir/"labels"

    for img_path in original_images.iterdir():
        label_path = (original_labels/img_path.name).with_suffix(".txt")
        print(f"processing {label_path}")

        img = cv2.imread(str(img_path))
        img_height, img_width = img.shape[:2]

        classes, bboxes = read_yolo_annotation(label_path)
        xyxy_bboxes = xywhn2xyxy(bboxes, w=img_width, h=img_height)

        w_ranges = get_split_ranges(img_width, num_splits=4, split_dim=640)
        h_ranges = get_split_ranges(img_height, num_splits=2, split_dim=640)

        save_image_crops(img, img_path, target_images, w_ranges, h_ranges)
        save_labels_crops(classes, xyxy_bboxes, label_path, target_labels, w_ranges, h_ranges)


if __name__ == "__main__":
    main()
