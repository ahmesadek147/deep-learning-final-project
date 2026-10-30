"""Command-line entry point for classification experiments."""
import argparse
from pathlib import Path

from src.predict import predict_image
from src.training.run_training import run_training
from src.training.experiment_config import EXPERIMENTS, load_experiment_config
from src.training.search import run_search
from src.evaluation import run_final_evaluation


def load_part_configs(path: Path, part: str) -> list[tuple[str, dict]]:
    """Load one or both parts from the single shared experiment YAML."""
    parts = EXPERIMENTS if part == "both" else (part,)
    return [(name, load_experiment_config(path, name)) for name in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Learning Final Project CLI")
    parser.add_argument("command", choices=["train", "search", "predict", "evaluate"])
    parser.add_argument("--task", choices=["classification", "colorization", "caption"], default="classification")
    parser.add_argument("--part", choices=["p4", "p5", "both"], help="Experiment selection for train/search.")
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"), help="YAML experiment configuration.")
    parser.add_argument("--image", type=str, help="Input image for prediction.")
    parser.add_argument("--checkpoint", type=Path, help="Selected metadata-bearing checkpoint for prediction.")
    parser.add_argument("--checkpoint-p4", type=Path, help="Best P4 checkpoint for final evaluation.")
    parser.add_argument("--checkpoint-p5", type=Path, help="Best P5 checkpoint for final evaluation.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"), help="Final evaluation output directory.")
    parser.add_argument("--gradcam-dir", type=Path, default=Path("outputs/gradcam"), help="Grad-CAM output directory for P5 evaluation.")
    parser.add_argument("--epochs", type=int, help="Override epochs for a direct smoke/train run.")
    parser.add_argument("--max-train-samples", type=int, help="Limit train samples for a direct smoke/train run.")
    parser.add_argument("--max-val-samples", type=int, help="Limit validation samples for a direct smoke/train run.")
    parser.add_argument("--split-strategy", choices=["legacy_random", "stratified_group"], help="Explicit train/validation split strategy for training.")
    parser.add_argument("--split-version", help="Version label for an explicit split manifest.")
    parser.add_argument("--split-manifest", type=Path, help="Validated train/validation split manifest.")
    parser.add_argument("--checkpoint-namespace", help="Separate checkpoint namespace for a training run.")
    args = parser.parse_args()
    if args.command == "predict":
        if not args.image:
            parser.error("Prediction requires --image")
        if not Path(args.image).is_file():
            parser.error(f"Image not found: {args.image}")
        if args.checkpoint is None:
            parser.error("Prediction requires --checkpoint")
        result = predict_image(image_path=args.image, task=args.task, checkpoint_path=args.checkpoint)
        print(f"Predicted class: {result['class_id']} ({result['confidence'] * 100:.2f}%)")
        return
    if args.command == "evaluate":
        if args.part not in {"p4", "p5", "both"}:
            parser.error("Evaluation requires --part p4, p5, or both")
        try:
            document_configs = load_part_configs(args.config, args.part)
            checkpoints = {}
            if args.part in {"p4", "both"}:
                if args.checkpoint_p4 is None: parser.error("--checkpoint-p4 is required for p4 evaluation")
                checkpoints["p4"] = str(args.checkpoint_p4)
            if args.part in {"p5", "both"}:
                if args.checkpoint_p5 is None: parser.error("--checkpoint-p5 is required for p5 evaluation")
                checkpoints["p5"] = str(args.checkpoint_p5)
            # Common settings are identical by construction; evaluate uses the
            # selected part's shared sections and does not tune on test data.
            config = document_configs[0][1]
            config["_mlflow_by_part"] = {name: value["mlflow"] for name, value in document_configs}
            print(run_final_evaluation(config, checkpoints, args.output_dir, args.gradcam_dir))
        except (OSError, ValueError, RuntimeError) as error:
            parser.error(str(error))
        return
    if args.task != "classification":
        parser.error("Only classification training is implemented.")
    if args.part is None:
        parser.error("Training and search require --part p4, p5, or both")
    try:
        part_configs = load_part_configs(args.config, args.part)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for part, config in part_configs:
        if args.command == "search" and any(value is not None for value in (args.epochs, args.max_train_samples, args.max_val_samples)):
            parser.error("--epochs and sample limits apply only to direct train smoke runs, not searches")
        if args.command == "train":
            if args.epochs is not None:
                if args.epochs < 1:
                    parser.error("--epochs must be positive")
                config["training"]["epochs"] = args.epochs
            if args.max_train_samples is not None:
                config.setdefault("data", {})["max_train_samples"] = args.max_train_samples
            if args.max_val_samples is not None:
                config.setdefault("data", {})["max_val_samples"] = args.max_val_samples
            if args.split_strategy is not None:
                config.setdefault("data", {})["split_strategy"] = args.split_strategy
            if args.split_version is not None:
                config.setdefault("data", {})["split_version"] = args.split_version
            if args.split_manifest is not None:
                config.setdefault("data", {})["split_manifest"] = str(args.split_manifest)
            if args.checkpoint_namespace is not None:
                config.setdefault("checkpoint", {})["namespace"] = args.checkpoint_namespace
        if args.command == "train":
            print({part: run_training(config)})
        else:
            print({part: run_search(config, strategy="grid" if part == "p4" else "random")})


if __name__ == "__main__":
    main()
