"""
Evaluate a trained YOLOv8 model on multiple test datasets (clean + degraded)
and export precision/recall results to a CSV file.

Usage:
    python scripts/test.py [--model MODEL_PATH] [--output OUTPUT_CSV] [--attention ATTENTION]

Available attention modules (via --attention):
    GAM       - Global Attention Module
    ECA       - Efficient Channel Attention
    RCBAM - Residual Block with CBAM
    SA        - Shuffle Attention
    none      - Standard YOLOv8 without attention (default)
"""

import argparse
import csv
import os
import sys
import shutil
import pdb
import yaml
# Ensure the local ultralytics package (with custom attention modules) is imported
# instead of any pip-installed version
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO

# Default test dataset names
DEFAULT_DATA_SETS = [
    "clean",
    "noise_2.5_percent",
    "noise_5.0_percent",
    "noise_10.0_percent",
    "noise_15.0_percent",
    "noise_20.0_percent",
    "noise_25.0_percent",
    "noise_30.0_percent",
    "blur_2.5_percent",
    "blur_10.0_percent",
    "blur_15.0_percent",
    "blur_20.0_percent",
    "blur_25.0_percent",
    "blur_30.0_percent",
]

# Available attention module YAML configs
ATTENTION_MODELS = {
    "none": "yolov8n.pt",          # Standard YOLOv8 without attention
    "GAM": "yolov8_GAM.yaml",      # Global Attention Module
    "ECA": "yolov8_ECA.yaml",      # Efficient Channel Attention
    "RCBAM": "yolov8_RCBAM.yaml",  # Residual Block with CBAM
    "SA": "yolov8_SA.yaml",        # Shuffle Attention
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on multiple test datasets")
    parser.add_argument(
        "--model",
        type=str,
        default='/apdcephfs_fsgm/share_304156246/xmudongwang/codebase/zq/YOLOv8-Crack-Detection/results_attention/degraded_data_RCBAM_multi_view_0.1/weights/best.pt',
        help="Path to trained model weights (default: auto-detected from --attention)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="Path to the test_datasets directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="none",
        choices=list(ATTENTION_MODELS.keys()),
        help=f"Attention module used during training: {list(ATTENTION_MODELS.keys())} (default: GAM)",
    )
    parser.add_argument(
        "--name-prefix",
        type=str,
        default='eval_degraded_data_RCBAM_multi_view_0.1',
        help="Name prefix for test result directories (default: auto from --attention)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Get project root directory (parent of scripts/)
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Auto-generate name prefix from attention module if not specified
    if args.name_prefix is None:
        if args.attention == "none":
            args.name_prefix = "test_results_50_epochs"
        else:
            args.name_prefix = f"test_results_{args.attention}"

    # Default model path: look for best.pt under results_attention/<attention_name>/weights/
    if args.model is None:
        if args.attention == "none":
            model_path = os.path.join(project_root, "results/clean_data_50_epochs/weights/best.pt")
        else:
            model_path = os.path.join(
                project_root,
                f"results_attention/clean_data_{args.attention}/weights/best.pt",
            )
    else:
        model_path = args.model

    # pdb.set_trace()
    # Default test datasets directory
    if args.test_dir is None:
        datasets_directory = os.path.join(project_root, "test_datasets")
    else:
        datasets_directory = args.test_dir

    # Default output CSV path
    if args.output is None:
        output_file = os.path.join(
            project_root,
            f"results_attention/{args.name_prefix}/results.csv",
        )
    else:
        output_file = args.output

    print(f"Project root:     {project_root}")
    print(f"Model:            {model_path}")
    print(f"Attention module: {args.attention}")
    print(f"Test datasets:    {datasets_directory}")
    print(f"Output CSV:       {output_file}")
    print()

    # Load the trained model
    model = YOLO(model_path)

    # Build dataset paths
    data_sets = DEFAULT_DATA_SETS
    data_sets_path = [os.path.join(datasets_directory, ds, "data.yaml") for ds in data_sets]

    # Lists to store results
    set_recall = []
    set_precision = []
    set_f1 = []
    set_map50 = []
    set_map50_95 = []
    set_num_images = []

    # Evaluate the model on each test set
    for i in range(len(data_sets_path)):
        save_path = os.path.join(project_root, f"results_attention/{args.name_prefix}")
        os.makedirs(save_path, exist_ok=True)

        # Copy test.py to output directory for reproducibility
        if i == 0:
            script_path = os.path.abspath(__file__)
            shutil.copy2(script_path, os.path.join(save_path, "test.py"))
            print(f"Copied test.py to: {save_path}/test.py")

        # pdb.set_trace()
        results = model.val(data=data_sets_path[i], save=True, project=save_path, name=data_sets[i])

        # Get mean results for precision and recall
        mean_precision, mean_recall, map50, map50_95 = results.mean_results()

        # Count the number of images in this test set
        with open(data_sets_path[i], 'r') as f:
            data_yaml = yaml.safe_load(f)
        val_path = data_yaml.get('val', data_yaml.get('test', ''))
        # Resolve relative path based on data.yaml location
        if not os.path.isabs(val_path):
            val_path = os.path.join(os.path.dirname(data_sets_path[i]), val_path)
        val_path = os.path.normpath(val_path)
        IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        num_images = len([f for f in os.listdir(val_path)
                         if os.path.splitext(f)[1].lower() in IMG_EXTS]) if os.path.isdir(val_path) else 0

        # Calculate F1-Score
        if mean_precision + mean_recall > 0:
            f1_score = 2 * mean_precision * mean_recall / (mean_precision + mean_recall)
        else:
            f1_score = 0.0

        # Store the metrics in the lists
        set_recall.append(mean_recall)
        set_precision.append(mean_precision)
        set_f1.append(f1_score)
        set_map50.append(map50)
        set_map50_95.append(map50_95)
        set_num_images.append(num_images)

        # Print the results
        print(f"{data_sets[i]} ({num_images} images) - Recall: {set_recall[i]:.4f}, Precision: {set_precision[i]:.4f}, F1: {set_f1[i]:.4f}, mAP50: {set_map50[i]:.4f}, mAP50-95: {set_map50_95[i]:.4f}")

    # Create the directory for output CSV if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Calculate simple average metrics across all test sets
    avg_recall = sum(set_recall) / len(set_recall)
    avg_precision = sum(set_precision) / len(set_precision)
    avg_f1 = sum(set_f1) / len(set_f1)
    avg_map50 = sum(set_map50) / len(set_map50)
    avg_map50_95 = sum(set_map50_95) / len(set_map50_95)

    # Calculate weighted average metrics (weighted by number of images)
    total_images = sum(set_num_images)
    if total_images > 0:
        w_avg_recall = sum(r * n for r, n in zip(set_recall, set_num_images)) / total_images
        w_avg_precision = sum(p * n for p, n in zip(set_precision, set_num_images)) / total_images
        w_avg_f1 = sum(f * n for f, n in zip(set_f1, set_num_images)) / total_images
        w_avg_map50 = sum(m * n for m, n in zip(set_map50, set_num_images)) / total_images
        w_avg_map50_95 = sum(m * n for m, n in zip(set_map50_95, set_num_images)) / total_images
    else:
        w_avg_recall = avg_recall
        w_avg_precision = avg_precision
        w_avg_f1 = avg_f1
        w_avg_map50 = avg_map50
        w_avg_map50_95 = avg_map50_95

    # Print average results
    print(f"\n{'='*60}")
    print(f"Simple Average across all {len(data_sets)} test sets:")
    print(f"  Recall:    {avg_recall:.4f}")
    print(f"  Precision: {avg_precision:.4f}")
    print(f"  F1-Score:  {avg_f1:.4f}")
    print(f"  mAP50:     {avg_map50:.4f}")
    print(f"  mAP50-95:  {avg_map50_95:.4f}")
    print(f"\nWeighted Average (by num images, total={total_images}):")
    print(f"  Recall:    {w_avg_recall:.4f}")
    print(f"  Precision: {w_avg_precision:.4f}")
    print(f"  F1-Score:  {w_avg_f1:.4f}")
    print(f"  mAP50:     {w_avg_map50:.4f}")
    print(f"  mAP50-95:  {w_avg_map50_95:.4f}")
    print(f"{'='*60}")

    # Write the results to a CSV file
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "NumImages", "Recall", "Precision", "F1-Score", "mAP50", "mAP50-95"])  # Write the header
        for i in range(len(data_sets)):
            writer.writerow([data_sets[i], set_num_images[i], set_recall[i], set_precision[i], set_f1[i], set_map50[i], set_map50_95[i]])
        # Write simple average row
        writer.writerow(["Simple_Average", total_images, avg_recall, avg_precision, avg_f1, avg_map50, avg_map50_95])
        # Write weighted average row
        writer.writerow(["Weighted_Average", total_images, w_avg_recall, w_avg_precision, w_avg_f1, w_avg_map50, w_avg_map50_95])

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()