"""
YOLOv11n train on Cityscapes (pedestrian & cyclist)

"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO

CLASS_MAP = {
    "person": ("pedestrian", 0),
    "rider":  ("cyclist",    1),
}

def parse_args():
    print("Parsing arguments...")
    p = argparse.ArgumentParser("Train YOLOv11n on Cityscapes")
    p.add_argument("--root",   type=Path, default=Path("dataset"),
                   help="Cityscapes folder (images/, annotations/)")
    p.add_argument("--data",   type=str,  default="dataset.yaml",
                   help="dataset.yaml")
    p.add_argument("--epochs", type=int,  default=50)
    p.add_argument("--imgsz",  type=int,  default=640)
    p.add_argument("--batch",  type=int,  default=16)
    p.add_argument("--weights",type=str,  default="yolov11n.pt")
    p.add_argument("--name",   type=str,  default="yolov11n_cityscapes")
    return p.parse_args()

def convert_cityscapes(root: Path):
    """
    Convert Cityscapes annotations to YOLO format (labels/*.txt).
    """
    ann_root = root / "annotations"
    img_root = root / "images"
    lbl_root = root / "labels"

    if lbl_root.exists() and any(lbl_root.rglob('*.txt')):
        return  

    print("converting annotations of Cityscapes to YOLO")
    for split in ("train", "val", "test"):
        for city in (ann_root / split).glob("*"):
            if not city.is_dir(): 
                continue
           
            for jf in city.glob("*_gtFine_polygons.json"):     
                img_name = jf.name.replace("_gtFine_polygons.json", "_leftImg8bit.png")
                img_path = img_root / split / city.name / img_name
                
                if not img_path.exists(): 
                    continue

                w, h = Image.open(img_path).size
                with open(jf) as f: 
                    data = json.load(f)

                lines = []
                for obj in data["objects"]:
                    if obj["label"] not in CLASS_MAP: 
                        continue

                    _, cls_id = CLASS_MAP[obj["label"]]
                    xs = [p[0] for p in obj["polygon"]]
                    ys = [p[1] for p in obj["polygon"]]
                    xmin, xmax = min(xs), max(xs)
                    ymin, ymax = min(ys), max(ys)
                    cx, cy = (xmin+xmax)/2/w, (ymin+ymax)/2/h
                    bw, bh = (xmax-xmin)/w, (ymax-ymin)/h
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                if lines:
                    out_dir = lbl_root / split / city.name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    (out_dir / img_name.replace(".png", ".txt")).write_text("\n".join(lines))
    print("finished converting")

def main():
    args = parse_args()
    convert_cityscapes(args.root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = YOLO(args.weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=device,
        pretrained=True,
        optimizer="auto",
    )

    best = Path(results.save_dir) / "best_yolov11.pt"
    
if __name__ == "__main__":
    main()