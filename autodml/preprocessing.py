import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
from utils.exception import PreprocessingError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer


class Preprocessor:
    def __init__(self, df, target_column):
        self.scaler = StandardScaler()
        self.df = df
        self.target = target_column
        self.feature_types = None

    def validate(self):
        logger = get_logger(__name__)
        logger.info("Starting data validation")

        try:
            if self.df is None:
                logger.error("No Data Found")
                raise ValueError(
                    "Data Error: No data was provided or file failed to load."
                )

            if not isinstance(self.df, pd.DataFrame):
                logger.error("Not a Pandas Dataframe")
                raise ValueError(
                    f"Type Error: Expected pandas DataFrame, got {type(self.df)}."
                )

            if self.df.empty:
                logger.error("Dataset Empty")
                raise ValueError("Data Error: The provided dataset is empty (0 rows).")

            if len(self.df.columns) < 2:
                logger.error(f"Dataset only has {len(self.df.columns)}")
                raise ValueError(
                    f"Structure Error: Dataset only has {len(self.df.columns)} column. "
                    "AutoML requires at least one feature and one target column."
                )

            if self.df.isnull().all().all():
                logger.error("All columns empty")
                raise ValueError(
                    "Data Error: All columns in the dataset are entirely null/empty."
                )

            if self.target not in self.df.columns:
                logger.error("Target Not Found")
                raise ValueError(
                    f"Target Error: Column '{self.target}' not found in dataset. "
                    f"Available columns: {list(self.df.columns)}"
                )

            if self.df[self.target].isnull().all():
                logger.error("Target Column Empty")
                raise ValueError(
                    f"Target Error: The target column '{self.target}' is 100% null. "
                    "The model has nothing to learn from."
                )

        except Exception as e:
            logger.exception("Error while Validating Dataset")
            raise PreprocessingError(
                message="Failed during Dataset Validation", details=str(e)
            )

        print("Data Validation Passed! Proceeding to preprocessing...")
        logger.info("Validation Completed")
        return True

    def remove_unwanted_columns(self) -> pd.DataFrame:
        logger = get_logger(__name__)
        try:
            logger.info("Starting Unwanted Column Removal")

            columns_to_drop = []

            for col in self.df.columns:
                if col.lower().startswith("unnamed"):
                    logger.debug(f"Dropping column '{col}' (Unnamed column)")
                    columns_to_drop.append(col)
                    continue

                if self.df[col].nunique(dropna=True) <= 1:
                    logger.debug(f"Dropping column '{col}' (Constant column)")
                    columns_to_drop.append(col)
                    continue

                if pd.api.types.is_integer_dtype(self.df[col]):
                    if (
                        self.df[col]
                        .sort_values()
                        .reset_index(drop=True)
                        .equals(pd.Series(range(len(self.df))))
                    ):
                        logger.debug(f"Dropping column '{col}' (Index-like column)")
                        columns_to_drop.append(col)
                        continue

            if columns_to_drop:
                logger.info(f"Dropping columns: {columns_to_drop}")
                self.df = self.df.drop(columns=columns_to_drop)

            logger.info("Unwanted column removal completed.")

            return self.df

        except Exception as e:
            logger.exception("Error while removing unwanted columns.")
            raise PreprocessingError(
                message="Failed during unwanted column removal", details=str(e)
            )

    def detect_feature_types(
        self,
        cat_threshold: int = 20,
        text_length_threshold: int = 30,
        id_unique_ratio: float = 0.95,
    ) -> dict:
        logger = get_logger(__name__)
        try:
            logger.info("Starting Automatic Feature Type Detection")

            feature_types = {
                "numerical": [],
                "categorical": [],
                "boolean": [],
                "datetime": [],
                "text": [],
                "id": [],
                "constant": [],
                "all_null": [],
            }

            n_rows = len(self.df)

            for col in self.df.columns:
                series = self.df[col]
                non_null = series.dropna()
                unique_count = non_null.nunique()

                if series.isnull().all():
                    feature_types["all_null"].append(col)
                    continue

                if unique_count <= 1:
                    feature_types["constant"].append(col)
                    continue

                if series.dtype == "bool" or set(non_null.unique()).issubset(
                    {0, 1, True, False}
                ):
                    feature_types["boolean"].append(col)
                    continue

                if pd.api.types.is_datetime64_any_dtype(series):
                    feature_types["datetime"].append(col)
                    continue

                if series.dtype == "object":
                    try:
                        parsed = pd.to_datetime(
                            non_null, errors="coerce", infer_datetime_format=True
                        )
                        success_ratio = parsed.notna().sum() / len(non_null)
                        if success_ratio > 0.8:
                            feature_types["datetime"].append(col)
                            continue

                    except Exception as e:
                        raise e

                if pd.api.types.is_numeric_dtype(series):
                    if unique_count <= cat_threshold:
                        feature_types["categorical"].append(col)
                    else:
                        if unique_count / n_rows > id_unique_ratio:
                            feature_types["id"].append(col)
                        else:
                            feature_types["numerical"].append(col)
                    continue

                if series.dtype == "object":
                    avg_length = non_null.astype(str).map(len).mean()

                    if unique_count / n_rows > id_unique_ratio:
                        feature_types["id"].append(col)

                    elif avg_length > text_length_threshold:
                        feature_types["text"].append(col)

                    else:
                        feature_types["categorical"].append(col)

                    continue

                feature_types["categorical"].append(col)
            self.feature_types = feature_types
            return feature_types

        except Exception as e:
            logger.exception("Error during feature type detection.")
            raise PreprocessingError(str(e))

    def Problem_detection(self):
        logger = get_logger(__name__)
        try:
            target_series = self.df[self.target].dropna()

            if target_series.empty:
                raise ValueError(
                    f"Target column '{self.target}' contains only null values."
                )

            if target_series.nunique() == 1:
                raise ValueError(
                    f"Target column '{self.target}' has only one unique value cannot be determine problem type."
                )

            if target_series.nunique() < 20 and target_series.dtype != "float":
                return "Classification"

            if pd.api.types.is_numeric_dtype(target_series):
                return "Regression"

            raise ValueError(
                "Unable to determine problem type: unexpected target data characteristics."
            )
        except Exception as e:
            logger.exception("Error during Problem Type detection.")
            raise PreprocessingError(str(e))

    def missing_value_handler(self):
        logger = get_logger(__name__)
        logger.info("Starting Missing Values Handling")
        missing_value_report = {}

        logger.info("Generating Missing Values Report")

        try:
            for i in self.df.columns:
                missing_value_report[i] = self.df[i].isna().sum()

            numerical_imputer = SimpleImputer(strategy="mean")
            categorical_imputer = SimpleImputer(strategy="most_frequent")

            for i in self.feature_types["categorical"]:
                if missing_value_report[i] > 0:
                    logger.info(f"Missing Values Found in {i} Column")
                    self.df[i] = categorical_imputer.fit_transform(self.df[i])
                    logger.info("Missing Values Imputed Into the Column {i}")
                else:
                    pass

            for i in self.feature_types["numerical"]:
                if missing_value_report[i] > 0:
                    logger.info(f"Missing Values Found in {i} Column")
                    self.df[i] = numerical_imputer.fit_transform(self.df[i])
                    logger.info("Missing Values Imputed Into the Column {i}")
                else:
                    pass

        except Exception as e:
            logger.exception("Error during Missing Value Handling")
            raise PreprocessingError(str(e))

        return self.df, missing_value_report

    def duplicate_handling(self):
        logger = get_logger(__name__)

        logger.info("Starting Duplicate Values Handling")

        try:
            duplicates = self.df.duplicated().sum()

            logger.info(f"{duplicates} Duplicate Values Found in the Dataset")

            if duplicates > 0:
                self.df = self.df.drop_duplicates()
                logger.info("Succesfully removed Duplicated")
            else:
                logger.info("No Duplicates Found")

        except Exception as e:
            logger.exception("Error during Duplicate Value Handling")
            raise PreprocessingError(str(e))

        return self.df

    def skewness_handling(self):
        logger = get_logger(__name__)
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        skewness = {}
        logger.info("Starting Skewness Handling")
        try:
            for i in self.feature_types["numerical"]:
                skewness[i] = self.df[i].skew()
                if self.df[i].skew() > 0.5 or self.df[i].skew() > -0.5:
                    self.df[i] = transformer.fit_transform(self.df[[i]])
                else:
                    pass

        except Exception as e:
            logger.exception("Error during Skewness Handling")
            raise PreprocessingError(str(e))

        logger.info("Skewness Handling Completed")
        return self.df, skewness

    def handle_outliers(self, method="auto", iqr_multiplier=1.5, z_threshold=3):
        logger = get_logger(__name__)
        logger.info("Starting Ouliers Handling")

        try:
            df = self.df.copy()
            report = {}

            numeric_cols = self.feature_types["numerical"]

            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    continue

                series = df[col].dropna()

                if len(series) < 10:
                    continue

                skew = series.skew()
                n = len(series)

                if method == "auto":
                    if n < 500:
                        chosen_method = "iqr"
                    elif abs(skew) > 1:
                        chosen_method = "iqr"
                    else:
                        chosen_method = "zscore"

                else:
                    chosen_method = method

                if chosen_method == "iqr":
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1

                    lower = Q1 - iqr_multiplier * IQR
                    upper = Q3 + iqr_multiplier * IQR

                    before = ((df[col] < lower) | (df[col] > upper)).sum()

                    df[col] = np.clip(df[col], lower, upper)

                    report[col] = {
                        "method": "IQR Capping",
                        "outliers_capped": int(before),
                    }

                elif chosen_method == "zscore":
                    mean = series.mean()
                    std = series.std()

                    lower = mean - z_threshold * std
                    upper = mean + z_threshold * std

                    before = ((df[col] < lower) | (df[col] > upper)).sum()

                    df[col] = np.clip(df[col], lower, upper)

                    report[col] = {
                        "method": "Z-score Capping",
                        "outliers_capped": int(before),
                    }

                logger.info("Completed Outlier Handling")

        except Exception as e:
            logger.exception("Error during Outlier Handling")
            raise PreprocessingError(str(e))

        self.df = df
        return df, report

    def process(self):
        print("Preprocessing Data...")

        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = pd.read_csv("temp\Student_performance.csv")
    target = "Performance Index"
    prep = Preprocessor(df, target_column=target)
    validation = prep.validate()
    problem = prep.Problem_detection()
    cleaned = prep.remove_unwanted_columns()
    features = prep.detect_feature_types()
    df, _ = prep.missing_value_handler()
    df, skew = prep.skewness_handling()
    df = prep.duplicate_handling()
    df, report = prep.handle_outliers()

    print(features)
    print(validation)
    print(problem)
    print(skew)
    print(report)
