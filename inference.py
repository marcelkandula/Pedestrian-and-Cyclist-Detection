"""
Pedestrian & Cyclist detection with YOLOv11 nano

Usage examples:
    python detect.py                     # webcam (default)
    python detect.py --source video --input data/test.mp4
    python detect.py --source video --input data/test.mp4 --output results/out.mp4

Arguments:
  --source   camera | video   (default: camera)
  --input    path to input video (required if source==video)
  --output   path to save annotated video (only if source==video)
  --model    path to .pt weights (default: yolov11n.pt)
  --conf     confidence threshold (default: 0.5)
"""

import argparse
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def parse_args():
    parser = argparse.ArgumentParser(description="Pedestrian and Cyclist detection with YOLOv11 nano")
    parser.add_argument("--source", choices=["camera", "video"], default="camera",
                        help="Input source: 'camera' or 'video' (default: camera)")
    parser.add_argument("--input", type=str, default="dataset/input.mp4",
                        help="Path to input video file if source is 'video'")
    parser.add_argument("--output", type=str, default="output/output.mp4",
                        help="Path to save annotated video (only if source is 'video')")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    return parser.parse_args()


def open_source(src_type: str, input_path: str | None):
    """Return an opened cv2.VideoCapture depending on source type."""
    if src_type == "camera":
        return cv2.VideoCapture(0)
    if input_path is None:
        raise ValueError("Argument --input is required when --source video")
    if not Path(input_path).is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    return cv2.VideoCapture(str(input_path))


def prepare_writer(cap: cv2.VideoCapture, output_path: str):
    """Create a VideoWriter that mirrors the properties of the capture."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def main():
    args = parse_args()
    print(f"Using device: {DEVICE}")

    # Load model (Ultralytics will download if missing)
    model = YOLO(args.model)

    cap = open_source(args.source, args.input)
    if not cap.isOpened():
        print("Unable to open video source")
        return

    writer = None
    if args.source == "video" and args.output:
        writer = prepare_writer(cap, args.output)
        print(f"Saving output to: {args.output}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = model.predict(frame,
                                    conf=args.conf,
                                    classes=[0, 1],  # 0='person', 1='bicycle' (COCO)
                                    device=DEVICE)[0]

            # Draw bounding boxes
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = results.names.get(cls_id, str(cls_id))

                color = (0, 255, 0) if cls_id == 0 else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf_val:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if writer is not None:
                writer.write(frame)

            cv2.imshow("YOLOv11 Pedestrian & Cyclist Detection", frame)
            if cv2.waitKey(1) == 27:  # Esc key
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
