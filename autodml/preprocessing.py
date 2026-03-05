import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
from utils.exception import PreprocessingError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score


class Preprocessor:
    def __init__(self, df, target_column, scale_features=True):
        self.scaler = StandardScaler()
        self.df = df
        self.target = target_column
        self.scale_features = scale_features
        self.feature_types = None
        self.problem_type = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def validate(self):
        try:
            if self.df is None:
                raise ValueError(
                    "Data Error: No data was provided or file failed to load."
                )

            if not isinstance(self.df, pd.DataFrame):
                raise ValueError(
                    f"Type Error: Expected pandas DataFrame, got {type(self.df)}."
                )

            if self.df.empty:
                raise ValueError("Data Error: The provided dataset is empty (0 rows).")

            if len(self.df.columns) < 2:
                raise ValueError(
                    f"Structure Error: Dataset only has {len(self.df.columns)} column. "
                    "AutoML requires at least one feature and one target column."
                )

            if self.df.isnull().all().all():
                raise ValueError(
                    "Data Error: All columns in the dataset are entirely null/empty."
                )

            if self.target not in self.df.columns:
                raise ValueError(
                    f"Target Error: Column '{self.target}' not found in dataset. "
                    f"Available columns: {list(self.df.columns)}"
                )

            if self.df[self.target].isnull().all():
                raise ValueError(
                    f"Target Error: The target column '{self.target}' is 100% null. "
                    "The model has nothing to learn from."
                )

        except Exception as e:
            raise PreprocessingError(
                message="Failed during Dataset Validation", details=str(e)
            )

        print("Data Validation Passed! Proceeding to preprocessing...")
        return True

    def remove_unwanted_columns(self) -> pd.DataFrame:
        try:
            columns_to_drop = []

            for col in self.df.columns:
                if col.lower().startswith("unnamed"):
                    columns_to_drop.append(col)
                    continue

                if self.df[col].nunique(dropna=True) <= 1:
                    columns_to_drop.append(col)
                    continue

                if pd.api.types.is_integer_dtype(self.df[col]):
                    if (
                        self.df[col]
                        .sort_values()
                        .reset_index(drop=True)
                        .equals(pd.Series(range(len(self.df))))
                    ):
                        columns_to_drop.append(col)
                        continue

            if columns_to_drop:
                self.df = self.df.drop(columns=columns_to_drop)
            return self.df

        except Exception as e:
            raise PreprocessingError(
                message="Failed during unwanted column removal", details=str(e)
            )

    def detect_feature_types(
        self,
        cat_threshold: int = 20,
        text_length_threshold: int = 30,
        id_unique_ratio: float = 0.95,
    ) -> dict:
        try:
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
            raise PreprocessingError(str(e))

    def Problem_detection(self):
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
                c = "Classification"
                self.problem_type = c
                return c

            if pd.api.types.is_numeric_dtype(target_series):
                r = "Regression"
                self.problem_type = r
                return r

            raise ValueError(
                "Unable to determine problem type: unexpected target data characteristics."
            )
        except Exception as e:
            raise PreprocessingError(str(e))

    def missing_value_handler(self):
        missing_value_report = {}

        try:
            for i in self.df.columns:
                missing_value_report[i] = self.df[i].isna().sum()

            numerical_imputer = SimpleImputer(strategy="mean")
            categorical_imputer = SimpleImputer(strategy="most_frequent")

            for i in self.feature_types["categorical"]:
                if missing_value_report[i] > 0:
                    self.df[i] = categorical_imputer.fit_transform(self.df[[i]])
                else:
                    pass

            for i in self.feature_types["numerical"]:
                if missing_value_report[i] > 0:
                    self.df[i] = numerical_imputer.fit_transform(self.df[[i]])
                else:
                    pass

        except Exception as e:
            raise PreprocessingError(str(e))

        return self.df, missing_value_report

    def duplicate_handling(self):
        try:
            duplicates = self.df.duplicated().sum()

            if duplicates > 0:
                self.df = self.df.drop_duplicates()
            else:
                pass

        except Exception as e:
            raise PreprocessingError(str(e))

        return self.df

    def skewness_handling(self):
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        skewness = {}
        try:
            for i in self.feature_types["numerical"]:
                skewness[i] = self.df[i].skew()
                if self.df[i].skew() > 0.5 or self.df[i].skew() > -0.5:
                    self.df[i] = transformer.fit_transform(self.df[[i]])
                else:
                    pass

        except Exception as e:
            raise PreprocessingError(str(e))

        return self.df, skewness

    def handle_outliers(self, method="auto", iqr_multiplier=1.5, z_threshold=3):
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

        except Exception as e:
            raise PreprocessingError(str(e))

        self.df = df
        return df, report

    def extract_datetime_features(
        self,
        drop_original: bool = True,
        add_cyclical: bool = True,
        add_duration: bool = True,
    ) -> pd.DataFrame:
        try:
            df = self.df.copy()
            datetime_cols = self.feature_types["datetime"]

            for col in datetime_cols:
                df[col] = pd.to_datetime(df[col], errors="coerce")

                if df[col].isnull().all():
                    continue
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_quarter"] = df[col].dt.quarter
                df[f"{col}_weekofyear"] = df[col].dt.isocalendar().week.astype("float")

                if not (df[col].dt.hour == 0).all():
                    df[f"{col}_hour"] = df[col].dt.hour
                    df[f"{col}_minute"] = df[col].dt.minute

                if add_duration:
                    min_date = df[col].min()
                    df[f"{col}_days_since"] = (df[col] - min_date).dt.days

                if add_cyclical:
                    df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12)
                    df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12)

                    df[f"{col}_dow_sin"] = np.sin(
                        2 * np.pi * df[f"{col}_dayofweek"] / 7
                    )
                    df[f"{col}_dow_cos"] = np.cos(
                        2 * np.pi * df[f"{col}_dayofweek"] / 7
                    )

                if drop_original:
                    df.drop(columns=[col], inplace=True)
            self.df = df
            return df
        except Exception as e:
            raise PreprocessingError(
                message="Error Caused WHile Selecting Datetime Features ",
                details=str(e),
            )

    def encoding(self):
        try:
            categorical = self.feature_types["categorical"]
            ids = self.feature_types["id"]

            for i in categorical:
                if i not in self.df.columns:
                    continue

                if not (
                    self.df[i].dtype == "object" or str(self.df[i].dtype) == "category"
                ):
                    continue

                n_unique = self.df[i].nunique()

                if n_unique == 2:
                    enc = LabelEncoder()
                    self.df[i] = enc.fit_transform(self.df[i])
                if n_unique > 2 and n_unique <= 10:
                    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                    self.df[i] = enc.fit_transform(self.df[[i]])
                if n_unique > 10:
                    enc = CountFrequencyEncoder(encoding_method="frequency")
                    self.df[i] = enc.fit_transform(self.df[[i]])

            for i in ids:
                enc = LabelEncoder()
                self.df[i] = enc.fit_transform(self.df[i])

        except Exception as e:
            raise PreprocessingError(message="Error While Encoding", details=str(e))

        self.x = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]

        return self.df

    def scaling(self):
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=0.2
            )
            scaler = StandardScaler()
            if self.scale_features == True:
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
            else:
                pass
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test

            return x_train, x_test, y_train, y_test

        except Exception as e:
            raise PreprocessingError(
                message="Error Caused While Scaling", details=str(e)
            )

    def dimensionality_reduction(self):
        try:
            if self.df.shape[1] > 250:
                pca = PCA(n_components=10)
                self.x_train = pca.fit_transform(self.x_train)
                self.x_test = pca.transform(self.x_test)
            else:
                pass
            return self.x_train, self.x_test

        except Exception as e:
            raise PreprocessingError(
                message="Error Caused While Dimensionality Reduction", details=str(e)
            )

    def process(self):
        print("Preprocessing Data...")

        self.validate()
        self.remove_unwanted_columns()
        self.detect_feature_types()
        self.Problem_detection()
        self.missing_value_handler()
        self.duplicate_handling()
        self.skewness_handling()
        self.handle_outliers()
        self.extract_datetime_features()
        self.encoding()
        self.scaling()
        self.dimensionality_reduction()

        return self.x_train, self.x_test, self.y_train, self.y_test


if __name__ == "__main__":
    df = pd.read_csv("temp/amazon_sales_dataset.csv")
    target = "Close"
    prep = Preprocessor(df, target_column=target)
    x_train, x_test, y_train, y_test = prep.process()
    model = LinearRegression()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(prep.problem_type)
    print(r2_score(pred, y_test))
