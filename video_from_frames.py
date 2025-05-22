"""
Create a video (.mp4) from a folder of image frames.

Usage:
    python video_from_frames.py --input PATH_TO_FRAMES_DIR --output OUTPUT_VIDEO.mp4
"""

import os
import argparse
import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a video file from a directory of image frames.'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to directory containing image frames'
    )
    parser.add_argument(
        '--output', '-o', default="output/output.mp4",
        required=False,
        help='Path to output video file (e.g., output.mp4)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    frames_dir = args.input
    output_path = args.output
    fps = 30.0  # default frames per second

    # Supported image extensions
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    # List and sort frame files
    files = [f for f in os.listdir(frames_dir)
             if os.path.splitext(f)[1].lower() in img_exts]
    if not files:
        print(f"No image files found in {frames_dir} with extensions: {img_exts}")
        return
    files.sort()

    first_frame = cv2.imread(os.path.join(frames_dir, files[0]))
    if first_frame is None:
        print(f"Failed to read first frame: {files[0]}")
        return
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for fname in files:
        path = os.path.join(frames_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: could not read {fname}, skipping.")
            continue
        # Resize if needed
        if img.shape[1] != width or img.shape[0] != height:
            img = cv2.resize(img, (width, height))
        writer.write(img)

    writer.release()
    print(f"Video successfully saved to {output_path}")


if __name__ == '__main__':
    main()
