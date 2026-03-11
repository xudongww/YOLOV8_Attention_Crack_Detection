"""
Train a YOLOv8n model on the clean crack detection dataset.

Usage:
    python train.py [--dataset DATASET_YAML] [--epochs EPOCHS] [--model MODEL] [--name NAME] [--attention ATTENTION]

    # Semi-supervised training (Pseudo-Label + EMA Teacher):
    python train.py --semi --unlabeled_dir /path/to/unlabeled/images [--semi_rounds 2] [--semi_conf 0.7]

Available attention modules (via --attention):
    RCBAM - Residual Block with CBAM
    none      - Standard YOLOv8 without attention (default)
"""

import argparse
import os
import sys
import shutil

# Ensure the local ultralytics package (with custom attention modules) is imported
# instead of any pip-installed version
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO


# Available attention module YAML configs
ATTENTION_MODELS = {
    "none": "yolov8n.pt",          # Standard YOLOv8 without attention
    "RCBAM": "yolov8_RCBAM.yaml",  # Residual Block with CBAM
}


def train(model, dataset, epochs, save_path, name, lr0=0.01, batch=16, inter_scale=0.0, intra_scale=0.0, multi_view=0.0):
    """
    Train the YOLO model with the specified dataset and number of epochs.

    Args:
        model: The YOLO model to be trained.
        dataset: Path to the dataset configuration file.
        epochs: Number of epochs to train the model.
        save_path: Directory to save training results.
        name: Name of the training run.
        lr0: Initial learning rate.
        batch: Batch size for training.
        inter_scale: Inter-scale consistency loss weight (0.0 to disable).
        intra_scale: Intra-scale consistency loss weight (0.0 to disable).
        multi_view: Multi-view consistency loss weight (0.0 to disable).
    """
    # Create the output directory and copy train.py into it
    output_dir = os.path.join(save_path, name)
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.abspath(__file__)
    shutil.copy2(script_path, os.path.join(output_dir, "train.py"))
    print(f"Copied train.py to: {output_dir}/train.py")

    _ = model.train(
        data=dataset,
        epochs=epochs,
        save=True,
        project=save_path,
        name=name,
        lr0=lr0,
        batch=batch,
        inter_scale=inter_scale,
        intra_scale=intra_scale,
        multi_view=multi_view,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 model on crack detection dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        # default='./zq/YOLOv8-Crack-Detection/training_cracks/degraded_dataset/data.yaml',
        default=None,
        help="Path to data.yaml (default: <project_root>/training_cracks/DawgSurfaceCracks/data.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs (default: 50)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pretrained model weights (.pt) or model config (.yaml). "
             "If not specified, will be determined by --attention flag.",
    )
    parser.add_argument(
        "--attention",
        type=str,
        default="RCBAM",
        choices=list(ATTENTION_MODELS.keys()),
        help=f"Attention module to use: {list(ATTENTION_MODELS.keys())} (default: none)",
    )
    # clean_data_RCBAM
    parser.add_argument("--name", type=str, default="clean_data_RCBAM_multi_view_0.1", help="Training run name")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (default: 0.01)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training (default: 16)")
    parser.add_argument(
        "--inter_scale",
        type=float,
        default=0.0,
        help="Inter-scale consistency loss weight. Set > 0 to enable self-supervised loss (e.g. 0.1). Default: 0.0 (disabled)",
    )
    parser.add_argument(
        "--intra_scale",
        type=float,
        default=0.0,
        help="Intra-scale consistency loss weight. Set > 0 to enable (e.g. 0.1). Default: 0.0 (disabled)",
    )
    parser.add_argument(
        "--multi_view",
        type=float,
        default=0.1,
        help="Multi-view consistency loss weight. Set > 0 to enable (e.g. 0.1). Default: 0.0 (disabled)",
    )

    # ----- Semi-supervised training args -----
    parser.add_argument(
        "--semi", action="store_true", default=False,
        help="Enable semi-supervised training with Pseudo-Label + EMA Teacher",
    )
    parser.add_argument(
        "--unlabeled_dir", type=str, default=None,
        help="Path to directory containing unlabeled images (required if --semi is set)",
    )
    parser.add_argument(
        "--semi_rounds", type=int, default=2,
        help="Number of self-training rounds after supervised warm-up (default: 2)",
    )
    parser.add_argument(
        "--semi_conf", type=float, default=0.7,
        help="Confidence threshold for pseudo-label filtering (default: 0.7)",
    )
    parser.add_argument(
        "--semi_epochs", type=int, default=None,
        help="Epochs for each self-training round (default: same as --epochs)",
    )
    parser.add_argument(
        "--semi_lr", type=float, default=None,
        help="Learning rate for self-training rounds (default: half of --lr)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))

    # Default dataset path
    if args.dataset is None:
        dataset = os.path.join(project_root, "training_cracks/DawgSurfaceCracks/data.yaml")
    else:
        dataset = args.dataset

    # Save path for results
    save_path = os.path.join(project_root, "results_attention")

    # Determine model path
    if args.model is not None:
        model_path = args.model
    else:
        model_name = ATTENTION_MODELS[args.attention]
        if model_name.endswith(".yaml"):
            # Look for the YAML config in the local ultralytics cfg directory
            model_path = os.path.join(
                project_root, "ultralytics", "cfg", "models", "v8", model_name
            )
        else:
            model_path = model_name  # e.g. "yolov8n.pt"

    # =========================================================================
    # Semi-supervised training mode
    # =========================================================================
    if args.semi:
        from train_semi import generate_pseudo_labels, create_merged_dataset

        if args.unlabeled_dir is None:
            args.unlabeled_dir = os.path.join(project_root, "training_cracks", "unlabeled", "images")
        if args.semi_epochs is None:
            args.semi_epochs = args.epochs
        if args.semi_lr is None:
            args.semi_lr = args.lr * 0.5

        if not os.path.isdir(args.unlabeled_dir):
            print(f"ERROR: Unlabeled image directory not found: {args.unlabeled_dir}")
            print("Please provide --unlabeled_dir or create the directory with unlabeled images.")
            sys.exit(1)

        print(f"{'='*60}")
        print(f"Semi-Supervised Training: Pseudo-Label + EMA Teacher")
        print(f"{'='*60}")
        print(f"  Labeled dataset:   {dataset}")
        print(f"  Unlabeled images:  {args.unlabeled_dir}")
        print(f"  Model:             {model_path}")
        print(f"  Attention:         {args.attention}")
        print(f"  Warm-up epochs:    {args.epochs}")
        print(f"  Self-train epochs: {args.semi_epochs}")
        print(f"  Self-train rounds: {args.semi_rounds}")
        print(f"  Pseudo-label conf: {args.semi_conf}")
        print(f"  Warm-up LR:        {args.lr}")
        print(f"  Self-train LR:     {args.semi_lr}")
        print(f"{'='*60}\n")

        # Round 0: Supervised warm-up
        round_name = f"{args.name}_round0_supervised"
        print(f"\n>>> Round 0: Supervised warm-up ({args.epochs} epochs)")
        model = YOLO(model_path)
        train(model, dataset, args.epochs, save_path, round_name,
              lr0=args.lr, batch=args.batch,
              inter_scale=args.inter_scale, intra_scale=args.intra_scale,
              multi_view=args.multi_view)

        best_model_path = os.path.join(save_path, round_name, "weights", "best.pt")
        if not os.path.exists(best_model_path):
            print(f"ERROR: best.pt not found at {best_model_path}")
            sys.exit(1)

        semi_work_dir = os.path.join(save_path, f"{args.name}_semi_workdir")
        os.makedirs(semi_work_dir, exist_ok=True)

        # Self-training rounds
        for round_idx in range(1, args.semi_rounds + 1):
            print(f"\n{'#'*60}")
            print(f"# Self-Training Round {round_idx}/{args.semi_rounds}")
            print(f"{'#'*60}")

            # Generate pseudo-labels
            pseudo_dir = os.path.join(semi_work_dir, f"pseudo_round{round_idx}")
            os.makedirs(pseudo_dir, exist_ok=True)
            n_pseudo = generate_pseudo_labels(
                best_model_path, args.unlabeled_dir, pseudo_dir, args.semi_conf)

            if n_pseudo == 0:
                print(f"WARNING: No pseudo-labels. Try lowering --semi_conf.")
                continue

            # Merge datasets
            merged_dir = os.path.join(semi_work_dir, f"merged_round{round_idx}")
            os.makedirs(merged_dir, exist_ok=True)
            merged_yaml = os.path.join(merged_dir, "data.yaml")
            create_merged_dataset(dataset, pseudo_dir, merged_yaml)

            # Train on merged dataset
            round_name = f"{args.name}_round{round_idx}_semi"
            model = YOLO(best_model_path)
            train(model, merged_yaml, args.semi_epochs, save_path, round_name,
                  lr0=args.semi_lr, batch=args.batch,
                  inter_scale=args.inter_scale, intra_scale=args.intra_scale,
                  multi_view=args.multi_view)

            new_best = os.path.join(save_path, round_name, "weights", "best.pt")
            if os.path.exists(new_best):
                best_model_path = new_best

        print(f"\nSemi-supervised training complete! Final model: {best_model_path}")
        return

    # =========================================================================
    # Normal supervised training mode (original behavior)
    # =========================================================================
    print(f"Project root:     {project_root}")
    print(f"Dataset:          {dataset}")
    print(f"Epochs:           {args.epochs}")
    print(f"Model:            {model_path}")
    print(f"Attention module: {args.attention}")
    print(f"Learning rate:    {args.lr}")
    print(f"Batch size:       {args.batch}")
    print(f"Inter-scale loss: {args.inter_scale}{'  (enabled)' if args.inter_scale > 0 else '  (disabled)'}")
    print(f"Intra-scale loss: {args.intra_scale}{'  (enabled)' if args.intra_scale > 0 else '  (disabled)'}")
    print(f"Multi-view loss:  {args.multi_view}{'  (enabled)' if args.multi_view > 0 else '  (disabled)'}")
    print(f"Save path:        {save_path}/{args.name}")

    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Train the model
    train(model, dataset, args.epochs, save_path, args.name, lr0=args.lr, batch=args.batch, inter_scale=args.inter_scale, intra_scale=args.intra_scale, multi_view=args.multi_view)

    print("Training complete!")


if __name__ == "__main__":
    main()
