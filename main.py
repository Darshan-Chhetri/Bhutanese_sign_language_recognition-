import sys
import os

# Ensure src module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from extract_landmarks import extract_landmarks
from train import train_model
from inference import run_inference

def run_pipeline(force_retrain=False):
    print("========================================")
    print("   BSL DIGIT RECOGNITION PIPELINE      ")
    print("========================================\n")
    print("Press 'Ctrl + C' at any time to exit.\n")

    train_csv = "data/train_landmarks.csv"
    test_csv = "data/test_landmarks.csv"
    model_path = "models/bsl_digit_model.pkl"

    try:
        # Step 1: Extract landmarks only if CSVs don't exist
        if not (os.path.exists(train_csv) and os.path.exists(test_csv)) or force_retrain:
            print("---> Step 1/3: Extracting landmarks from images...")
            extract_landmarks("dataset/bsl/train", train_csv)
            extract_landmarks("dataset/bsl/test", test_csv)
        else:
            print("[✓] Step 1/3: Extracted landmarks found in 'data/'. Skipping image extraction.")

        # Step 2: Train model only if model file doesn't exist
        if not os.path.exists(model_path) or force_retrain:
            print("\n---> Step 2/3: Training Random Forest model...")
            train_model()
        else:
            print("[✓] Step 2/3: Trained model found in 'models/'. Skipping training.")

        # Step 3: Launch live camera
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
    # If you pass '--retrain' flag, it forces full re-extraction and re-training
    force = "--retrain" in sys.argv
    run_pipeline(force_retrain=force)