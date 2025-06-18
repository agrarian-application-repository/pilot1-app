import cv2
from pathlib import Path


def draw_yolo_annotations(image_path, label_path, class_names=None, color=(0, 255, 0)):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None
    h, w = img.shape[:2]

    if not label_path.exists():
        print("no labels")
        return img

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # skip incomplete lines

            cls, x_center, y_center, bw, bh = map(float, parts)
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            print("plotting")
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            if class_names:
                label = class_names[int(cls)]
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


def main(images_dir, labels_dir, output_dir, class_file=None):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = None
    if class_file and Path(class_file).exists():
        with open(class_file) as f:
            class_names = [line.strip() for line in f]

    for img_path in images_dir.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        label_path = labels_dir / (img_path.stem + ".txt")
        annotated_img = draw_yolo_annotations(img_path, label_path, class_names)

        if annotated_img is not None:
            out_path = output_dir / img_path.name
            cv2.imwrite(str(out_path), annotated_img)
            print(f"Saved: {out_path}")
        else:
            print(f"Skipped: {img_path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save YOLO-annotated images.")
    parser.add_argument("--images", required=True, help="Path to images folder")
    parser.add_argument("--labels", required=True, help="Path to YOLO labels folder")
    parser.add_argument("--output", required=True, help="Folder to save annotated images")
    parser.add_argument("--classes", default=None, help="Optional path to classes.txt")

    args = parser.parse_args()

    main(args.images, args.labels, args.output, args.classes)

# python projects/AGRARIAN_PROJECT/AGRARIAN/pyscripts/show_yolo_annotated_images.py --images annotated_test/images --labels annotated_test/labels --output annotated_test/annotated_images --classes annotated_test/classes.txt

