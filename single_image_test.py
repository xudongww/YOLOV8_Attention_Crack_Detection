"""
Single image evaluation & visualization for the trained YOLOv8 crack detection model.

Produces a 3-panel figure: Original Image | Ground Truth | Predictions.

Usage:
    python scripts/single_image_test.py [--model MODEL_PATH] [--image-dir DIR] [--label-dir DIR] [--index INDEX] [--conf CONF]
"""

import argparse
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO


def load_ground_truth(label_path):
    """
    Load ground truth bounding boxes from a YOLO-format label file.
    Supports polygon-format labels (> 4 coordinate values).

    Returns:
        List of (cls_id, x_center, y_center, width, height) in normalized coords.
    """
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    if len(coords) == 4:
                        # Standard YOLO bbox format: x_center y_center width height
                        x_center, y_center, width, height = coords
                    else:
                        # Polygon format: x1 y1 x2 y2 ... compute bounding box
                        xs = coords[0::2]  # All x coordinates
                        ys = coords[1::2]  # All y coordinates
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                    gt_boxes.append((cls_id, x_center, y_center, width, height))
    return gt_boxes


def visualize_single_image(model, image_path, label_path, save_path, conf=0.1, iou=0.7):
    """
    Run inference on a single image and visualize Original / GT / Predictions.
    """
    image_name = os.path.basename(image_path)

    # Run inference
    results = model.predict(source=image_path, conf=conf, iou=iou, verbose=False)
    result = results[0]

    # Load ground truth
    gt_boxes = load_ground_truth(label_path)

    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- Panel 1: Original Image ---
    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold")
    axes[0].axis("off")

    # --- Panel 2: Ground Truth ---
    axes[1].imshow(img)
    axes[1].set_title(f"Ground Truth ({len(gt_boxes)} objects)", fontsize=14, fontweight="bold")
    axes[1].axis("off")
    for cls_id, xc, yc, w, h in gt_boxes:
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        bw = w * img_w
        bh = h * img_h
        rect = patches.Rectangle(
            (x1, y1), bw, bh, linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1, y1 - 4, "crack", fontsize=8, color="lime",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1),
        )

    # --- Panel 3: Predictions ---
    pred_boxes = result.boxes
    num_preds = len(pred_boxes)
    axes[2].imshow(img)
    axes[2].set_title(f"Predictions ({num_preds} detections)", fontsize=14, fontweight="bold")
    axes[2].axis("off")

    if num_preds > 0:
        for box in pred_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf_val = box.conf[0].cpu().numpy()
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor="red", facecolor="none",
            )
            axes[2].add_patch(rect)
            axes[2].text(
                x1, y1 - 4, f"crack {conf_val:.2f}", fontsize=8, color="red",
                bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1),
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  Single Image Evaluation Summary")
    print(f"{'=' * 50}")
    print(f"  Image:        {image_name}")
    print(f"  Image size:   {img_w} x {img_h}")
    print(f"  GT objects:   {len(gt_boxes)}")
    print(f"  Predictions:  {num_preds}")
    if num_preds > 0:
        confs = pred_boxes.conf.cpu().numpy()
        print(f"  Conf range:   [{confs.min():.4f}, {confs.max():.4f}]")
        print(f"  Conf mean:    {confs.mean():.4f}")
    print(f"{'=' * 50}")
    print(f"\nVisualization saved to: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Single image evaluation for YOLOv8 crack detection")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model weights")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing test images")
    parser.add_argument("--label-dir", type=str, default=None, help="Directory containing label files")
    parser.add_argument("--index", type=int, default=20, help="Index of image to evaluate (default: 20)")
    parser.add_argument("--image", type=str, default=None, help="Direct path to a single image (overrides --image-dir and --index)")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold (default: 0.1)")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold (default: 0.7)")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Get project root directory (parent of scripts/)
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default model path
    if args.model is None:
        model_path = os.path.join(current_directory, "results/clean_data_50_epochs/weights/best.pt")
    else:
        model_path = args.model

    # Default image/label directories
    if args.image_dir is None:
        image_dir = os.path.join(current_directory, "test_datasets/clean/valid/images")
    else:
        image_dir = args.image_dir

    if args.label_dir is None:
        label_dir = os.path.join(current_directory, "test_datasets/clean/valid/labels")
    else:
        label_dir = args.label_dir

    # Default output path
    if args.output is None:
        output_path = os.path.join(current_directory, "results/single_image_eval.png")
    else:
        output_path = args.output

    # Determine image path
    if args.image is not None:
        image_path = args.image
        image_name = os.path.basename(image_path)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
    else:
        image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
        )
        if not image_files:
            print(f"No images found in {image_dir}")
            return

        print(f"Total images available: {len(image_files)}")
        image_index = min(args.index, len(image_files) - 1)
        image_name = image_files[image_index]
        image_path = os.path.join(image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

    print(f"Selected image: {os.path.basename(image_path)}")

    # Load model
    model = YOLO(model_path, verbose=False)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Run visualization
    visualize_single_image(model, image_path, label_path, output_path, conf=args.conf, iou=args.iou)


if __name__ == "__main__":
    main()
