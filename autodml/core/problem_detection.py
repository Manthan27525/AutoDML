import pandas as pd
import numpy as np


class ProblemDetector:
    @staticmethod
    def detect(df, target_column, threshold=0.05):
        """
        Detects if a target variable implies a Classification or Regression problem.

        Args:
            df (pd.DataFrame): The dataset.
            target_column (str): The name of the target variable.
            threshold (float): The ratio of unique values to total rows
                            above which an integer target is treated as regression.
        """
        if target_column not in df.columns:
            return "Error: Target column not found in DataFrame."

        target = df[target_column].dropna()
        unique_count = target.nunique()
        total_count = len(target)

        if (
            target.dtype == "object"
            or target.dtype.name == "category"
            or target.dtype == "bool"
        ):
            return "Classification"

        if np.issubdtype(target.dtype, np.floating):
            if unique_count <= 10:
                return "Classification (Low cardinality Float)"
            return "Regression"

        if np.issubdtype(target.dtype, np.integer):
            if unique_count <= 20 or (unique_count / total_count) < threshold:
                return "Classification"
            else:
                return "Regression"

        return "Unknown"


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    df = pd.read_csv(path)
    target = "Performance Index"
    print(ProblemDetector.detect(df, target))
