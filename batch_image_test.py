"""
Batch single-image evaluation for the trained YOLOv8 crack detection model.

Evaluates multiple images in a grid layout and produces a confidence distribution
analysis chart.

Usage:
    python scripts/batch_image_test.py [--model MODEL_PATH] [--num-samples N] [--conf CONF]
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
        List of (x_center, y_center, width, height) in normalized coords.
    """
    gt = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    coords = list(map(float, parts[1:]))
                    if len(coords) == 4:
                        # Standard YOLO bbox format: x_center y_center width height
                        xc, yc, bw, bh = coords
                    else:
                        # Polygon format: x1 y1 x2 y2 ... compute bounding box
                        xs = coords[0::2]
                        ys = coords[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        xc = (x_min + x_max) / 2
                        yc = (y_min + y_max) / 2
                        bw = x_max - x_min
                        bh = y_max - y_min
                    gt.append((xc, yc, bw, bh))
    return gt


def batch_evaluate(model, image_dir, label_dir, num_samples, save_dir, conf=0.25, iou=0.7):
    """
    Evaluate and visualize multiple images in a grid.
    Also produces a confidence distribution analysis.
    """
    # Get all image files
    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    )
    if not image_files:
        print(f"No images found in {image_dir}")
        return

    num_samples = min(num_samples, len(image_files))
    cols = 3
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)

    all_confs = []
    all_num_gt = []
    all_num_pred = []

    for idx in range(num_samples):
        row, col = divmod(idx, cols)
        ax = axes[row][col]

        img_name = image_files[idx]
        img_path = os.path.join(image_dir, img_name)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(label_dir, lbl_name)

        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Run inference
        res = model.predict(source=img_path, conf=conf, iou=iou, verbose=False)
        pred = res[0].boxes

        # Load GT (polygon format -> bounding box)
        gt = load_ground_truth(lbl_path)

        all_num_gt.append(len(gt))
        all_num_pred.append(len(pred))

        # Draw image
        ax.imshow(img)

        # Draw GT boxes (green)
        for xc, yc, bw, bh in gt:
            x1 = (xc - bw / 2) * w
            y1 = (yc - bh / 2) * h
            rect = patches.Rectangle(
                (x1, y1), bw * w, bh * h,
                linewidth=1.5, edgecolor="lime", facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)

        # Draw predictions (red)
        if len(pred) > 0:
            for box in pred:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val = box.conf[0].cpu().numpy()
                all_confs.append(conf_val)
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.5, edgecolor="red", facecolor="none",
                )
                ax.add_patch(rect)
                ax.text(
                    x1, y1 - 2, f"{conf_val:.2f}", fontsize=7, color="red",
                    bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=0.5),
                )

        short_name = img_name[:25] + "..." if len(img_name) > 25 else img_name
        ax.set_title(f"{short_name}\nGT: {len(gt)} | Pred: {len(pred)}", fontsize=10)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(num_samples, rows * cols):
        row, col = divmod(idx, cols)
        axes[row][col].axis("off")

    fig.suptitle(
        "Single Image Evaluation Grid (Green=GT, Red=Pred)",
        fontsize=16, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    grid_path = os.path.join(save_dir, "batch_single_image_eval.png")
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grid visualization saved to: {grid_path}")

    # ======================== Confidence Distribution ========================
    if all_confs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confidence histogram
        axes[0].hist(all_confs, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
        axes[0].axvline(
            np.mean(all_confs), color="red", linestyle="--", linewidth=1.5,
            label=f"Mean: {np.mean(all_confs):.3f}",
        )
        axes[0].set_xlabel("Confidence Score", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].set_title("Prediction Confidence Distribution", fontsize=14, fontweight="bold")
        axes[0].legend(fontsize=11)
        axes[0].grid(axis="y", alpha=0.3)

        # GT vs Predictions bar chart
        x_labels = [f"Img {i}" for i in range(num_samples)]
        x = np.arange(num_samples)
        bar_width = 0.35
        axes[1].bar(x - bar_width / 2, all_num_gt, bar_width, label="Ground Truth", color="lime", edgecolor="gray")
        axes[1].bar(x + bar_width / 2, all_num_pred, bar_width, label="Predictions", color="tomato", edgecolor="gray")
        axes[1].set_xlabel("Image Index", fontsize=12)
        axes[1].set_ylabel("Number of Objects", fontsize=12)
        axes[1].set_title("GT vs Predictions per Image", fontsize=14, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(x_labels, fontsize=8)
        axes[1].legend(fontsize=11)
        axes[1].grid(axis="y", alpha=0.3)

        plt.tight_layout()
        conf_path = os.path.join(save_dir, "confidence_analysis.png")
        plt.savefig(conf_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Confidence analysis saved to: {conf_path}")

    # Print summary
    print(f"\nBatch evaluation complete for {num_samples} images.")
    print(f"Total GT objects: {sum(all_num_gt)}, Total predictions: {sum(all_num_pred)}")
    if all_confs:
        print(f"Overall confidence - Mean: {np.mean(all_confs):.4f}, Std: {np.std(all_confs):.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch image evaluation for YOLOv8 crack detection")
    parser.add_argument("--model", type=str, default=None, help="Path to trained model weights")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing test images")
    parser.add_argument("--label-dir", type=str, default=None, help="Directory containing label files")
    parser.add_argument("--num-samples", type=int, default=9, help="Number of images to evaluate (default: 9)")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold (default: 0.7)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output images")
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

    # Default output directory
    if args.output_dir is None:
        save_dir = os.path.join(current_directory, "results")
    else:
        save_dir = args.output_dir

    print(f"Project root:   {current_directory}")
    print(f"Model:          {model_path}")
    print(f"Image dir:      {image_dir}")
    print(f"Label dir:      {label_dir}")
    print(f"Num samples:    {args.num_samples}")
    print(f"Conf threshold: {args.conf}")
    print(f"Output dir:     {save_dir}")
    print()

    # Load model
    model = YOLO(model_path, verbose=False)

    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Run batch evaluation
    batch_evaluate(model, image_dir, label_dir, args.num_samples, save_dir, conf=args.conf, iou=args.iou)


if __name__ == "__main__":
    main()
