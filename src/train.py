import os
import joblib

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def run_training():
    # 1. Load handwritten digits dataset
    digits = load_digits()

    X = digits.data
    y = digits.target

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(set(y))}")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Train model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy:.4f}")

    # 5. Save model
    os.makedirs("model", exist_ok=True)

    joblib.dump(
        model,
        "model/digit_model.pkl"
    )

    print("Saved trained model to model/digit_model.pkl")


if __name__ == "__main__":
    run_training()