
import argparse
import yaml
from src.training.run_training import run_training
from pathlib import Path
from src.predict import predict_image


def main():
    parser = argparse.ArgumentParser(
        description="Deep Learning Final Project CLI"
    )

    parser.add_argument(
        "command",
        choices=["train", "predict"],
        help="Choose train or predict"
    )

    parser.add_argument(
        "--task",
        choices=["classification", "colorization", "caption"],
        required=True,
        help="Choose the task"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image"
    )

    args = parser.parse_args()

    print("Command:", args.command)
    print("Task:", args.task)
    print("Config:", args.config)
    print("Image:", args.image)


    if args.config:
        with open(args.config, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        print("\nConfiguration Loaded Successfully!")
        print("Model:", config["model"]["name"])
        print("Epochs:", config["training"]["epochs"])
        print("Batch Size:", config["training"]["batch_size"])
        print("Learning Rate:", config["training"]["learning_rate"])
        
        if args.command == "train":
            if args.task == "classification":
                run_training(config)
            else:
                print(
                    f"Training for {args.task} is not implemented yet."
                )

    if args.command == "predict":
        if not args.image:
            parser.error("Prediction requires --image")

        image_path = Path(args.image)

        if not image_path.is_file():
            parser.error(f"Image not found: {args.image}")

        result = predict_image(
            image_path=args.image,
            task=args.task
        )

        print("\nPrediction Completed!")
        print("Predicted Class:", result["class_id"])
        print("Confidence:", f'{result["confidence"] * 100:.2f}%')      

if __name__ == "__main__":
    main()