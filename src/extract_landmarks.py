import os
import cv2
import mediapipe as mp
import pandas as pd

mp_hands = mp.solutions.hands

def extract_landmarks(dataset_path, output_csv_path):
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    rows = []
    total_images = 0
    detected_images = 0

    if not os.path.exists(dataset_path):
        print(f"Error: Directory '{dataset_path}' not found.")
        return

    for label in sorted(os.listdir(dataset_path)):
        class_dir = os.path.join(dataset_path, label)
        if not os.path.isdir(class_dir):
            continue

        print(f"Processing digit '{label}' in {dataset_path}...")

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            frame = cv2.imread(img_path)

            if frame is None:
                continue

            total_images += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                detected_images += 1
                hand_landmarks = results.multi_hand_landmarks[0]
                wrist = hand_landmarks.landmark[0]

                # Normalize 21 coordinates relative to wrist position
                x_coords = [lm.x - wrist.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y - wrist.y for lm in hand_landmarks.landmark]
                z_coords = [lm.z - wrist.z for lm in hand_landmarks.landmark]

                row = x_coords + y_coords + z_coords + [str(label)]
                rows.append(row)

    # Construct columns: x0..x20, y0..y20, z0..z20, label
    columns = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)] + ["label"]
    df = pd.DataFrame(rows, columns=columns)

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    print(f"Done. Detected landmarks in {detected_images}/{total_images} images.")
    print(f"Saved: {output_csv_path}\n")

if __name__ == "__main__":
    extract_landmarks("dataset/bsl/train", "data/train_landmarks.csv")
    extract_landmarks("dataset/bsl/test", "data/test_landmarks.csv")