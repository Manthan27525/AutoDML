import pandas as pd

import pandas as pd
import numpy as np


class FeatureDetector:
    @staticmethod
    def auto_detect_feature_types(df, cat_unique_ratio=0.05, text_avg_len=30):
        """
        Automatically detect feature types: numeric, categorical, datetime, text
        """

        feature_types = {"numeric": [], "categorical": [], "datetime": []}

        # n_rows = df.shape[0]

        for col in df.columns:
            series = df[col]
            non_null = series.dropna()

            if len(non_null) == 0:
                feature_types["unknown"].append(col)
                continue

            # ===== NUMERIC CHECK =====
            if pd.api.types.is_numeric_dtype(series):
                feature_types["numeric"].append(col)
                continue

            # Try numeric conversion
            try:
                converted = pd.to_numeric(non_null, errors="raise")
                feature_types["numeric"].append(col)
                continue
            except:
                pass

            # ===== DATETIME CHECK =====
            if pd.api.types.is_datetime64_any_dtype(series):
                feature_types["datetime"].append(col)
                continue

            try:
                converted = pd.to_datetime(non_null, errors="raise")
                feature_types["datetime"].append(col)
                continue
            except:
                pass

            # ===== CATEGORICAL vs TEXT =====
            unique_ratio = non_null.nunique() / len(non_null)

            # Average text length
            avg_len = non_null.astype(str).str.len().mean()

            if unique_ratio < cat_unique_ratio:
                feature_types["categorical"].append(col)

            elif avg_len > text_avg_len:
                feature_types["text"].append(col)

            else:
                feature_types["categorical"].append(col)

        return feature_types


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    df = pd.read_csv(path)
    types = FeatureDetector.auto_detect_feature_types(df)
    print(types["numeric"])
    print(types["categorical"])
