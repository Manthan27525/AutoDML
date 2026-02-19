import pandas as pd
import numpy as np


class SkewnessAnalyzer:
    @staticmethod
    def Analyze(df, numeric_cols: list[str]):
        skew = {}
        for i in numeric_cols:
            skew[i] = df[i].skew()
        return skew


if __name__ == "__main__":
    from autodml.preprocessing.feature_types import FeatureDetector

    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    df = pd.read_csv(path)
    detector = FeatureDetector()
    col = detector.auto_detect_feature_types(df)["numeric"]
    skew = SkewnessAnalyzer.Analyze(df, col)
    print(skew)
