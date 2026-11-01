from pathlib import Path
import json
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "student-mat.csv"
MODEL_DIR = BASE_DIR / "model"

FEATURES = [
    "age", "studytime", "failures", "absences",
    "traveltime", "famsup", "paid", "higher", "internet"
]

def main():

    df = pd.read_csv(DATA_PATH, sep=";")

    print("Dataset loaded successfully!")
    print("Dataset Shape:", df.shape)

    df = df.drop_duplicates().dropna().copy()

    df["Performance"] = df["G3"].apply(
        lambda x: "Pass" if x >= 10 else "Fail"
    )

    X = df[FEATURES]
    y = df["Performance"]

    numerical_features = [
        "age", "studytime", "failures", "absences", "traveltime"
    ]

    categorical_features = [
        "famsup", "paid", "higher", "internet"
    ]

    preprocessor = ColumnTransformer([
        ("numeric", "passthrough", numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            random_state=42,
            max_depth=5
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, pos_label="Pass", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, pos_label="Pass", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, pos_label="Pass", zero_division=0
    )

    cm = confusion_matrix(
        y_test, predictions, labels=["Fail", "Pass"]
    )

    metrics = {
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": cm.tolist(),
        "total_students": len(df),
        "pass_count": int((y == "Pass").sum()),
        "fail_count": int((y == "Fail").sum()),
        "features": FEATURES
    }

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_DIR / "student_model.pkl")

    with open(MODEL_DIR / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nMODEL EVALUATION")
    print("----------------------")
    print("Accuracy:", metrics["accuracy"], "%")
    print("Precision:", metrics["precision"], "%")
    print("Recall:", metrics["recall"], "%")
    print("F1 Score:", metrics["f1_score"], "%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("Confusion Matrix:")
    print(cm)

    print("\nModel saved successfully!")
    print("Metrics saved successfully!")


if __name__ == "__main__":
    main()