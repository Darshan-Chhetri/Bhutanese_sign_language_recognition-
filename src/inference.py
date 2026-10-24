import os
import cv2
import joblib
import numpy as np
import mediapipe as mp

def run_inference():
    model_path = "models/bsl_digit_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found. Run train.py first.")
        return

    clf = joblib.load(model_path)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    print("Starting webcam... Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        predicted_digit = "None"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                wrist = hand_landmarks.landmark[0]
                x_coords = [lm.x - wrist.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y - wrist.y for lm in hand_landmarks.landmark]
                z_coords = [lm.z - wrist.z for lm in hand_landmarks.landmark]

                features = np.array(x_coords + y_coords + z_coords).reshape(1, -1)
                
                probabilities = clf.predict_proba(features)
                predicted_digit = clf.predict(features)[0]
                confidence = np.max(probabilities) * 100

        # UI Overlay
        cv2.putText(
            frame, 
            f"BSL Digit: {predicted_digit} ({confidence:.1f}%)", 
            (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2
        )

        cv2.imshow("BSL Digit Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()