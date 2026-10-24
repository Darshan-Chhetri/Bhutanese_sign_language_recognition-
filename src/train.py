import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    train_path = "data/train_landmarks.csv"
    test_path = "data/test_landmarks.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("CSV files not found. Run extract_landmarks.py first.")
        return

    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df.iloc[:, :-1].values, train_df.iloc[:, -1].values
    X_test, y_test = test_df.iloc[:, :-1].values, test_df.iloc[:, -1].values

    print(f"Training on {len(X_train)} samples | Testing on {len(X_test)} samples...")

    # Initialize and train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100

    print("\n--- Evaluation Results ---")
    print(f"Test Accuracy: {accuracy:.2f}%\n")
    print(classification_report(y_test, y_pred))

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/bsl_digit_model.pkl"
    joblib.dump(clf, model_save_path)
    print(f"Saved trained model to {model_save_path}")

if __name__ == "__main__":
    train_model()