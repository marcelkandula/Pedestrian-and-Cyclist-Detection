# Pedestrian and Cyclist Detection

A simple set of Python scripts for real-time pedestrian and cyclist detection, validation and traing using YOLOv11n and CityScapes dataset.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/marcelkandula/Pedestrian-and-Cyclist-Detection
   cd Pedestrian-and-Cyclist-Detection
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Scripts

### 1. `test.py`

Detect pedestrians and cyclists from a live camera or a video file.

**Usage**:

```bash
python test.py [--source {camera,video}] [--input <path>] [--output <path>] [--model <weights>.pt] [--conf <0.0–1.0>] 
```

* `--source`: `camera` (default) or `video`
* `--input`: Path to the input video file (required if `--source` is `video`)
* `--output`: Path to save the annotated output video
* `--weights`: Path to YOLOv11 nano weights (default: `yolov11n.pt`)
* `--conf`: Confidence threshold (default: `0.5`)

**Example**:

```bash
python test.py --source video --input input/test.mp4 --output output/out.mp4
```

### 2. `validate.py`

Validate yolo model using videos from /test_vids

**Usage**:

```bash
python validate.py --weights <YOLO_wegihts> --img-dir <frames_directory> --label-dir <label_directory> --save-dir <output_results>
```

### 3. `train.py`

Train yolo model by modifying dataset.yaml file

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
