from pathlib import Path
from datetime import datetime
import csv
import json
import joblib
import pandas as pd

from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "student_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
HISTORY_PATH = BASE_DIR / "history.csv"

FIELDS = [
    {
        "name": "age",
        "label": "Age",
        "type": "number",
        "min": 15,
        "max": 22,
        "help": "Student age (15–22)"
    },
    {
        "name": "studytime",
        "label": "Weekly Study Time",
        "type": "select",
        "options": [
            ("1", "Less than 2 hours"),
            ("2", "2–5 hours"),
            ("3", "5–10 hours"),
            ("4", "More than 10 hours")
        ],
        "help": "Select weekly study time"
    },
    {
        "name": "failures",
        "label": "Past Class Failures",
        "type": "number",
        "min": 0,
        "max": 3,
        "help": "Number of previous failures"
    },
    {
        "name": "absences",
        "label": "School Absences",
        "type": "number",
        "min": 0,
        "max": 93,
        "help": "Number of school absences"
    },
    {
        "name": "traveltime",
        "label": "Travel Time to School",
        "type": "select",
        "options": [
            ("1", "Less than 15 minutes"),
            ("2", "15–30 minutes"),
            ("3", "30 minutes–1 hour"),
            ("4", "More than 1 hour")
        ],
        "help": "Usual one-way travel time"
    },
    {
        "name": "famsup",
        "label": "Family Educational Support",
        "type": "select",
        "options": [("yes", "Yes"), ("no", "No")],
        "help": "Does the student receive family support?"
    },
    {
        "name": "paid",
        "label": "Paid Extra Classes",
        "type": "select",
        "options": [("yes", "Yes"), ("no", "No")],
        "help": "Does the student attend paid extra classes?"
    },
    {
        "name": "higher",
        "label": "Interested in Higher Education",
        "type": "select",
        "options": [("yes", "Yes"), ("no", "No")],
        "help": "Does the student want higher education?"
    },
    {
        "name": "internet",
        "label": "Internet Access at Home",
        "type": "select",
        "options": [("yes", "Yes"), ("no", "No")],
        "help": "Does the student have internet access?"
    }
]


def get_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as file:
            return json.load(file)

    return {}


@app.route("/")
def home():
    return render_template("index.html", metrics=get_metrics())


@app.route("/predict", methods=["GET", "POST"])
def predict():

    result = None
    error = None

    if request.method == "POST":

        try:
            values = {}

            for field in FIELDS:

                raw = request.form.get(field["name"], "").strip()

                if not raw:
                    raise ValueError(
                        f"Please enter {field['label']}."
                    )

                if field["type"] == "number":

                    value = int(raw)

                    if value < field["min"] or value > field["max"]:
                        raise ValueError(
                            f"{field['label']} must be between "
                            f"{field['min']} and {field['max']}."
                        )

                    values[field["name"]] = value

                else:
                    allowed = [
                        option[0] for option in field["options"]
                    ]

                    if raw not in allowed:
                        raise ValueError("Invalid selection.")

                    values[field["name"]] = raw

            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "Model not found. Run python train_model.py first."
                )

            model = joblib.load(MODEL_PATH)

            input_data = pd.DataFrame([values])

            prediction = model.predict(input_data)[0]

            probabilities = model.predict_proba(input_data)[0]
            class_names = list(model.classes_)

            score = round(
                float(probabilities[class_names.index(prediction)]) * 100,
                2
            )

            result = {
                "prediction": prediction,
                "score": score
            }

            new_file = not HISTORY_PATH.exists()

            with open(HISTORY_PATH, "a", newline="") as file:

                writer = csv.writer(file)

                if new_file:
                    writer.writerow([
                        "date_time", "prediction", "model_score"
                    ])

                writer.writerow([
                    datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    prediction,
                    score
                ])

        except Exception as e:
            error = str(e)

    return render_template(
        "predict.html",
        fields=FIELDS,
        result=result,
        error=error
    )


@app.route("/analytics")
def analytics():
    return render_template(
        "analytics.html",
        metrics=get_metrics()
    )


@app.route("/history")
def history():

    rows = []

    if HISTORY_PATH.exists():

        with open(HISTORY_PATH, newline="") as file:
            rows = list(csv.DictReader(file))

        rows.reverse()

    return render_template("history.html", rows=rows)


@app.route("/about")
def about():
    return render_template(
        "about.html",
        metrics=get_metrics()
    )


if __name__ == "__main__":
    app.run(debug=True)