import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_classification(output_folder):
    data = {
        "StudyHours": [8, 6, 9, 3, 5, 7, 4, 2, 10, 1, 6, 3, 8, 4, 7, 2, 9, 5, 1, 6],
        "Attendance": [92, 80, 95, 60, 75, 78, 65, 55, 96, 50, 82, 58, 90, 62, 85, 52, 93, 70, 45, 77],
        "AssignmentScore": [85, 72, 90, 45, 58, 66, 52, 40, 92, 35, 70, 48, 88, 50, 76, 38, 91, 60, 30, 68],
        "Result": ["Pass", "Pass", "Pass", "Fail", "Pass", "Pass", "Fail", "Fail", "Pass", "Fail",
                   "Pass", "Fail", "Pass", "Fail", "Pass", "Fail", "Pass", "Pass", "Fail", "Pass"]
    }

    df = pd.DataFrame(data)

    X = df[["StudyHours", "Attendance", "AssignmentScore"]]
    y = df["Result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=["Fail", "Pass"]
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Fail", "Actual Pass"],
        columns=["Predicted Fail", "Predicted Pass"]
    )

    report = classification_report(y_test, predictions)

    results = pd.DataFrame({
        "ActualResult": y_test,
        "PredictedResult": predictions
    })

    classification_text = f"""
Classification Evaluation Report
--------------------------------

Prediction Results
------------------
{results}

Accuracy
--------
Accuracy: {accuracy:.2f}
Accuracy Percentage: {accuracy * 100:.2f}%

Confusion Matrix
----------------
{matrix_df}

Classification Report
---------------------
{report}
"""

    print(classification_text)

    output_file = output_folder / "classification_evaluation.txt"

    with open(output_file, "w") as file:
        file.write(classification_text)


def evaluate_regression(output_folder):
    data = {
        "StudyHours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Score": [35, 45, 50, 55, 65, 70, 75, 85, 88, 95]
    }

    df = pd.DataFrame(data)

    X = df[["StudyHours"]]
    y = df["Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame({
        "StudyHours": X_test["StudyHours"],
        "ActualScore": y_test,
        "PredictedScore": predictions.round(2),
        "AbsoluteError": abs(y_test - predictions).round(2)
    })

    regression_text = f"""
Regression Evaluation Report
----------------------------

Prediction Results
------------------
{results}

Error Measurements
------------------
Mean Absolute Error: {mae:.2f}
Mean Squared Error: {mse:.2f}
Root Mean Squared Error: {rmse:.2f}
R-squared Score: {r2:.2f}
"""

    print(regression_text)

    output_file = output_folder / "regression_evaluation.txt"

    with open(output_file, "w") as file:
        file.write(regression_text)


def main():
    output_folder = Path("outputs")
    output_folder.mkdir(exist_ok=True)

    evaluate_classification(output_folder)
    evaluate_regression(output_folder)

    print("Evaluation reports saved in the outputs folder.")


main()
