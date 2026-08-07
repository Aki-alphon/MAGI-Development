"""
MAGI Vision — Demo Script
==========================
Quick demonstration of the training and prediction pipeline.

Usage:
    python demo.py --train       Run full training pipeline
    python demo.py --predict     Predict from sample image
    python demo.py --api         Start the FastAPI server
"""

import argparse
import sys


def run_training():
    """Execute the full MAGI training pipeline."""
    from magi_vision.pipeline.training_pipeline import MAGITrainPipeline

    print("\n🚀 Starting MAGI Vision Training Pipeline...\n")
    pipeline = MAGITrainPipeline()
    pipeline.run_pipeline()
    print("\n✅ Training complete!\n")


def run_prediction(image_path: str = None):
    """Run prediction on a sample image."""
    import cv2
    import json
    from magi_vision.pipeline.prediction_pipeline import MAGIPredictionPipeline

    predictor = MAGIPredictionPipeline()

    if image_path:
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Cannot read image: {image_path}")
            return

        print(f"\n🖼️  Analyzing: {image_path}")
        result = predictor.predict_single(img)
        print(json.dumps(result, indent=2))

        print(f"\n🌾 Canopy analysis:")
        canopy = predictor.predict_canopy(img)
        print(json.dumps(canopy, indent=2))
    else:
        print("No image specified. Use: python demo.py --predict path/to/image.jpg")


def run_api():
    """Start the FastAPI server."""
    import uvicorn
    from magi_vision.constants import APP_HOST, APP_PORT

    print(f"\n🌐 Starting MAGI Vision API at http://{APP_HOST}:{APP_PORT}")
    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAGI Vision Demo")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--predict", type=str, nargs="?", const="", help="Predict from image")
    parser.add_argument("--api", action="store_true", help="Start FastAPI server")

    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.predict is not None:
        run_prediction(args.predict if args.predict else None)
    elif args.api:
        run_api()
    else:
        parser.print_help()