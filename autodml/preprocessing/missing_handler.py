import pandas as pd
from sklearn.impute import SimpleImputer
from autodml.preprocessing.missing_analysis import MissingAnalyzer
from autodml.preprocessing.feature_types import FeatureDetector


class MissingValueHandler:
    @staticmethod
    def Impute(df):
        imputer1 = SimpleImputer(strategy="mean")
        imputer2 = SimpleImputer(strategy="most_frequent")
        missing = MissingAnalyzer.missing_cols(df)
        types = FeatureDetector.auto_detect_feature_types(df)
        for i in missing:
            if i in types["numeric"]:
                df[i] = imputer1.fit_transform(df[[i]]).flatten()
            elif i in types["categorical"]:
                df[i] = imputer2.fit_transform(df[[i]]).flatten()
            else:
                df[i] = imputer2.fit_transform(df[[i]]).flatten()
        return df, missing


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    df = pd.read_csv(path)
    print(df.isna().sum())
    df, missing = MissingValueHandler.Impute(df)
    print(missing)
    print(df.isna().sum())
