import sys
import os

# Ensure src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from extract_landmarks import extract_landmarks
from train import train_model
from inference import run_inference

def run_pipeline():
    print("----------------------------------------")
    print("   BSL DIGIT RECOGNITION PIPELINE      ")
    print("----------------------------------------\n")
    print("Press 'Ctrl + C' at any time to exit.\n")

    try:
        # Step 1: Extract landmark features from dataset images
        print("---> Step 1/3: Extracting landmarks from images...")
        extract_landmarks("dataset/bsl/train", "data/train_landmarks.csv")
        extract_landmarks("dataset/bsl/test", "data/test_landmarks.csv")

        # Step 2: Train the model on extracted landmarks
        print("\n---> Step 2/3: Training Random Forest model...")
        train_model()

        # Step 3: Launch webcam for real-time inference
        print("\n---> Step 3/3: Launching live webcam recognition...")
        run_inference()

    except KeyboardInterrupt:
        print("\n\n[!] Process interrupted by user (Ctrl + C). Cleaning up and exiting...")
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(0)

if __name__ == "__main__":
    run_pipeline()