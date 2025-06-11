import argparse
from pathlib import Path
import yaml

from ultralytics import YOLO  

def build_dataset_yaml(img_dir: Path, yaml_path: Path) -> None:

    data = {
        "path": str(img_dir.parent),
        "train": "images",
        "val": "images",
        "test": "images",
        "nc": 2,
        "names": ["pedestrian", "cyclist"],
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Validate YOLOv11n models")
    p.add_argument("--weights", required=True,
                   help="path to weights (.pt)")
    p.add_argument("--img-dir", default="test_vids/images",
                   help="images folder (images/1, images/2 …)")
    p.add_argument("--label-dir", default="test_vids/labels",
                   help="label folders (labels/1, labels/2 …)")
    p.add_argument("--save-dir", default="runs/test",
                   help="output folder for results/metrics")
    p.add_argument("--img-size", type=int, default=640,
                   help="input size")
    p.add_argument("--conf", type=float, default=0.3,
                   help="confidence threshold")
    args = p.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)


    dataset_yaml = save_dir / "dataset.yaml"
    build_dataset_yaml(Path(args.img_dir), dataset_yaml)


    model = YOLO(args.weights)


    results = model.val(
        data=str(dataset_yaml),
        imgsz=args.img_size,
        conf=args.conf,
        project=str(save_dir),
        name="val",
        half=False,
        save_json=True,
        classes=[0, 1]
    )

    box = results.box
    f1 = box.f1.mean()

    print("\n=== Metrics  ===")
    print(f"Precision        : {box.mp:.4f}")
    print(f"Recall           : {box.mr:.4f}")
    print(f"F1-score         : {f1:.4f}")
    print(f"mAP@0.50         : {box.map50:.4f}")
    print(f"mAP@0.50-0.95    : {box.map:.4f}")
    print(30 * "-")

if __name__ == "__main__":
    main()