"""
====================================================
Module : evaluate.py
Project: Airline Passenger Forecasting
Purpose: Evaluate LSTM Model Performance
====================================================
"""

import os
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from src.predict import Predictor


class Evaluator:
    """
    Evaluate model performance.
    """

    def __init__(self):
        pass

    def evaluate(self):

        # -----------------------------
        # Generate Predictions
        # -----------------------------
        predictor = Predictor()

        actual, predicted = predictor.predict()

        # -----------------------------
        # Calculate Metrics
        # -----------------------------
        mae = mean_absolute_error(
            actual,
            predicted
        )

        mse = mean_squared_error(
            actual,
            predicted
        )

        rmse = np.sqrt(mse)

        print("\n==============================")
        print("     MODEL EVALUATION")
        print("==============================")

        print(f"MAE  : {mae:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")

        # -----------------------------
        # Create Output Folder
        # -----------------------------
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # -----------------------------
        # Save Metrics
        # -----------------------------
        output_file = os.path.join(
            output_dir,
            "metrics.txt"
        )

        with open(output_file, "w") as file:

            file.write("Model Evaluation\n")
            file.write("=====================\n\n")

            file.write(f"MAE  : {mae:.4f}\n")
            file.write(f"MSE  : {mse:.4f}\n")
            file.write(f"RMSE : {rmse:.4f}\n")

        print(f"\nMetrics Saved Successfully.")
        print(f"Location : {output_file}")

        return mae, mse, rmse


if __name__ == "__main__":

    evaluator = Evaluator()

    mae, mse, rmse = evaluator.evaluate()