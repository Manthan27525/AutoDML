import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from autodml.preprocessing.feature_types import FeatureDetector


class Scaler:
    def Scale(df):
        df_scaled = df.copy()
        numeric_cols = FeatureDetector.auto_detect_feature_types(df_scaled)["numeric"]
        scaling_log = {}

        for col in numeric_cols:
            skewness = df[col].skew()

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            outlier_percentage = len(outliers) / len(df)

            if outlier_percentage > 0.05:
                scaler = RobustScaler()
                scaling_log[col] = "RobustScaler (High Outliers)"

            elif -0.5 < skewness < 0.5:
                scaler = StandardScaler()
                scaling_log[col] = "StandardScaler (Normal Distribution)"

            else:
                scaler = MinMaxScaler()
                scaling_log[col] = "MinMaxScaler (General/Skewed)"

            df_scaled[col] = scaler.fit_transform(df_scaled[[col]])

        return df_scaled, scaling_log


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    df = pd.read_csv(path)
    scaled_df = Scaler.Scale(df)
    print(scaled_df)
