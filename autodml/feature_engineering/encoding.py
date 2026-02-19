import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from autodml.preprocessing import feature_types


class Encoder:
    @staticmethod
    def CategoricalEncoder(df, threshold=10):
        df_encoded = df.copy()
        categorical_cols = feature_types.FeatureDetector.auto_detect_feature_types(
            df_encoded
        )["categorical"]

        encoding_log = {}

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        le = LabelEncoder()

        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()

            if unique_count <= threshold:
                encoding_log[col] = f"One-Hot ({unique_count} categories)"
                df_encoded[col] = ohe.fit_transform(df_encoded[[col]])

            else:
                encoding_log[col] = f"Label Encoded ({unique_count} categories)"
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        return df_encoded, encoding_log


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    df = pd.read_csv(path)
    df_encoded, _ = Encoder.CategoricalEncoder(df)
    print(df_encoded.head())
    print(_)
