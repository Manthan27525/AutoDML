import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
from utils.exception import PreprocessingError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = get_logger(__name__)


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


class Preprocessor:
    def __init__(self, df, target_column, scale_features=True):
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
        self.encoders = {}
        self.vectorizers = {}
        self.scaler = None
        self.input_features = None
        self.cat_imputer = None
        self.num_imputer = None
        self.skew_transformer = None
        self.pca = None

    def validate(self):
        logger.info("Starting Data Validation")
        try:
            logger.debug("Checking for error while loading data.")
            if self.df is None:
                logger.error("Data Error: No data was provided or file failed to load.")
                raise ValueError(
                    "Data Error: No data was provided or file failed to load."
                )

            logger.debug("Validating dataframe data type.")
            if not isinstance(self.df, pd.DataFrame):
                logger.error(
                    f"Type Error: Expected pandas DataFrame, got {type(self.df)}."
                )
                raise ValueError(
                    f"Type Error: Expected pandas DataFrame, got {type(self.df)}."
                )
            logger.debug("Checking if dataframe is empty.")
            if self.df.empty:
                logger.error("Data Error: The provided dataset is empty (0 rows).")
                raise ValueError("Data Error: The provided dataset is empty (0 rows).")

            logger.debug("Validating Dataset structure.")
            if len(self.df.columns) < 2:
                logger.error(
                    f"Structure Error: Dataset only has {len(self.df.columns)} column. "
                )
                raise ValueError(
                    f"Structure Error: Dataset only has {len(self.df.columns)} column. "
                    "AutoML requires at least one feature and one target column."
                )

            logger.debug("Checking for empty data columns.")
            if self.df.isnull().all().all():
                logger.error(
                    "Data Error: All columns in the dataset are entirely null/empty."
                )
                raise ValueError(
                    "Data Error: All columns in the dataset are entirely null/empty."
                )

            logger.debug("Validating target column.")
            if self.target not in self.df.columns:
                logger.error(
                    f"Target Error: Column '{self.target}' not found in dataset. "
                )
                raise ValueError(
                    f"Target Error: Column '{self.target}' not found in dataset. "
                    f"Available columns: {list(self.df.columns)}"
                )

            logger.debug("Checking for empty dataset.")
            if self.df[self.target].isnull().all():
                logger.error(
                    f"Target Error: The target column '{self.target}' is 100% null. "
                )
                raise ValueError(
                    f"Target Error: The target column '{self.target}' is 100% null. "
                    "The model has nothing to learn from."
                )

        except Exception as e:
            logger.error("Preprocessing Error: Failed during Dataset validation")
            raise PreprocessingError(
                message="Failed during Dataset Validation", details=str(e)
            )

        logger.info("Data Validation Completed.")
        return True

    def remove_unwanted_columns(self) -> pd.DataFrame:
        logger.info("Removing Unwanted Columns.")
        try:
            columns_to_drop = []

            for col in self.df.columns:
                logger.debug("Checking for unnamed columns.")
                if col.lower().startswith("unnamed"):
                    columns_to_drop.append(col)
                    logger.info(f"Unnamed columns found : {col} and Removed.")
                    continue

                logger.debug("Checking for constant columns")
                if self.df[col].nunique(dropna=True) <= 1:
                    columns_to_drop.append(col)
                    logger.info("Removing Constant Columns.")
                    continue

                logger.debug("Checking for Duplicate Index Column.")
                if pd.api.types.is_integer_dtype(self.df[col]):
                    if (
                        self.df[col]
                        .sort_values()
                        .reset_index(drop=True)
                        .equals(pd.Series(range(len(self.df))))
                    ):
                        columns_to_drop.append(col)
                        logger.info("Duplicate Index Removed.")
                        continue

            if columns_to_drop:
                self.df = self.df.drop(columns=columns_to_drop)

            logger.info("Unwanted Columns Removed.")
            return self.df

        except Exception as e:
            logger.error("Preprocessing Error : Failed during unwanted column removal")
            raise PreprocessingError(
                message="Failed during unwanted column removal", details=str(e)
            )

    def detect_feature_types(
        self,
        cat_threshold: int = 20,
        text_length_threshold: int = 30,
        id_unique_ratio: float = 0.95,
        sample_size: int = 5000,
    ) -> dict:
        logger.info("Starting Feature Types Detection.")
        try:
            df = self.df

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

            n_rows = len(df)

            for col in df.columns:
                series = df[col]

                logger.debug("Checking for Null Columns.")
                if series.isna().all():
                    feature_types["all_null"].append(col)
                    logger.warning("Null columns found !")
                    continue

                non_null = series.dropna()

                logger.debug("Checking for Constant Columns.")
                if non_null.nunique() <= 1:
                    feature_types["constant"].append(col)
                    logger.warning("Constant Columns Found !")
                    continue
                sample = non_null
                if len(non_null) > sample_size:
                    sample = non_null.sample(sample_size, random_state=42)

                unique_count = sample.nunique()

                if pd.api.types.is_bool_dtype(series):
                    feature_types["boolean"].append(col)
                    continue

                if pd.api.types.is_numeric_dtype(series):
                    unique_vals = set(sample.unique())

                    if unique_vals.issubset({0, 1}):
                        feature_types["boolean"].append(col)
                        continue

                if pd.api.types.is_datetime64_any_dtype(series):
                    feature_types["datetime"].append(col)
                    continue

                if series.dtype == "object":
                    parsed = pd.to_datetime(sample, errors="coerce")

                    success_ratio = parsed.notna().mean()

                    if success_ratio > 0.8:
                        feature_types["datetime"].append(col)
                        continue

                if pd.api.types.is_numeric_dtype(series):
                    if unique_count <= cat_threshold:
                        feature_types["categorical"].append(col)

                    elif unique_count / n_rows > id_unique_ratio:
                        feature_types["id"].append(col)

                    else:
                        feature_types["numerical"].append(col)

                    continue

                if series.dtype == "object" or pd.api.types.is_categorical_dtype(
                    series
                ):
                    avg_len = sample.astype(str).str.len().mean()

                    if unique_count / n_rows > id_unique_ratio:
                        feature_types["id"].append(col)

                    elif avg_len > text_length_threshold:
                        feature_types["text"].append(col)

                    else:
                        feature_types["categorical"].append(col)

                    continue

                feature_types["categorical"].append(col)

            self.feature_types = feature_types

            logger.info("Feature Detection Completed.")
            return feature_types

        except Exception as e:
            logger.error(str(e))
            raise PreprocessingError(str(e))

    def handling_textual_data(self):
        logger.info("Handling Textual Data")
        try:
            df = self.df
            text_cols = self.feature_types["text"]
            vectorizers = {}

            for col in text_cols:
                if col not in df.columns:
                    continue

                df[col] = df[col].astype(str).apply(preprocess_text)

                vectorizer = TfidfVectorizer(max_features=100)

                text_matrix = vectorizer.fit_transform(df[col].astype(str))
                df = df.reset_index(drop=True)
                text_df = pd.DataFrame(
                    text_matrix.toarray(),
                    columns=[f"{col}_tfidf_{i}" for i in range(text_matrix.shape[1])],
                )

                df = df.drop(columns=[col])
                vectorizers[col] = vectorizer
                df = pd.concat([df, text_df], axis=1)

            self.df = df
            self.vectorizers = vectorizers
            logger.info("Text Feature Handling Completed")

            return df

        except Exception as e:
            logger.error(str(e))
            raise PreprocessingError(
                message="Error While Handling Textual Data", details=str(e)
            )

    def Problem_detection(self):
        logger.info("Detecting Problem Type.")
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

            if target_series.dtype == "object":
                c = "Classification"
                self.problem_type = c
                return c

            if target_series.nunique() < 20 and target_series.dtype != "float":
                c = "Classification"
                self.problem_type = c
                return c

            if pd.api.types.is_numeric_dtype(target_series):
                r = "Regression"
                self.problem_type = r
                return r

            logger.info("Problem Detection Completed.")

            raise ValueError(
                "Unable to determine problem type: unexpected target data characteristics."
            )

        except Exception as e:
            logger.error(str(e))
            raise PreprocessingError(str(e))

    def missing_value_handler(self):
        missing_value_report = {}
        logger.info("Starting Missing Value Handling.")
        try:
            for i in self.df.columns:
                missing_value_report[i] = self.df[i].isna().sum()

            num_cols = self.feature_types["numerical"]
            if self.target in num_cols:
                num_cols.remove(self.target)
            if num_cols:
                self.num_imputer = SimpleImputer(strategy="mean")
                self.df[num_cols] = self.num_imputer.fit_transform(self.df[num_cols])

            cat_cols = self.feature_types["categorical"]
            if self.target in cat_cols:
                cat_cols.remove(self.target)
            if cat_cols:
                self.cat_imputer = SimpleImputer(strategy="most_frequent")
                self.df[cat_cols] = self.cat_imputer.fit_transform(self.df[cat_cols])
            else:
                pass

            id_cols = self.feature_types["id"]
            if id_cols:
                id_imputer = SimpleImputer(strategy="most_frequent")
                self.df[id_cols] = id_imputer.fit_transform(self.df[id_cols])

        except Exception as e:
            logger.error(str(e))
            raise PreprocessingError(str(e))

        logger.info("Missing Value Handling Completed.")
        return self.df, missing_value_report

    def duplicate_handling(self):
        logger.info("Initiating Duplicate Value Handling.")
        try:
            duplicates = self.df.duplicated().sum()

            if duplicates > 0:
                self.df = self.df.drop_duplicates()
            else:
                pass

        except Exception as e:
            logger.error(str(e))
            raise PreprocessingError(str(e))

        logger.info("Duplicate Value Handling Completed.")
        return self.df

    def skewness_handling(self):
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        skewness = {}
        logger.info("Starting Skew Data Handling.")
        try:
            for i in self.feature_types["numerical"]:
                skewness[i] = self.df[i].skew()
                if self.df[i].skew() > 0.5 or self.df[i].skew() > -0.5:
                    self.df[i] = transformer.fit_transform(self.df[[i]])
                    self.skew_transformer = transformer
                else:
                    pass

        except Exception as e:
            logger.error(str(e))
            raise PreprocessingError(str(e))

        logger.info("Skewness Handling Completed.")
        return self.df, skewness

    def handle_outliers(self, method="auto", iqr_multiplier=1.5, z_threshold=3):
        logger.info("Handling Outliers.")
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
            logger.error(str(e))
            raise PreprocessingError(str(e))

        logger.info("Ouliers Handling Completed.")
        self.df = df
        return df, report

    def extract_datetime_features(
        self,
        drop_original: bool = True,
        add_cyclical: bool = True,
        add_duration: bool = True,
    ) -> pd.DataFrame:
        logger.info("Extracting Datetime Features.")
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

            logger.info("Datetime Features Extracted.")
            self.df = df
            return df
        except Exception as e:
            logger.error(str(e))
            raise PreprocessingError(
                message="Error Caused WHile Selecting Datetime Features ",
                details=str(e),
            )

    def encoding(self):
        logger.info("Starting Encoding.")
        try:
            categorical = self.feature_types.get("categorical", [])
            ids = self.feature_types.get("id", [])

            for col in categorical + ids:
                if col == self.target or col not in self.df.columns:
                    continue

                self.df[col] = self.df[col].astype(str).str.strip()

                n_unique = self.df[col].nunique()

                if n_unique <= 10:
                    enc = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )

                    self.df[[col]] = enc.fit_transform(self.df[[col]])
                    self.encoders[col] = enc
                    logger.info(f"Encoded {col} using OrdinalEncoder")
                else:
                    enc = CountFrequencyEncoder(encoding_method="frequency")
                    self.df[[col]] = enc.fit_transform(self.df[[col]])
                    self.encoders[col] = enc
                    logger.info(f"Encoded {col} using FrequencyEncoder")

            if self.problem_type == "Classification" and self.target in self.df.columns:
                t_enc = LabelEncoder()
                self.df[self.target] = t_enc.fit_transform(
                    self.df[self.target].astype(str).str.strip()
                )
                self.encoders["_TARGET_"] = t_enc
                logger.info("Encoded Target using LabelEncoder")

            self.input_features = [c for c in self.df.columns if c != self.target]
            return self.df

        except Exception as e:
            logger.error(f"Encoding Error: {str(e)}")
            raise PreprocessingError(str(e))

    def scaling(self):
        logger.info("Starting Feature Scaling.")
        try:
            x_train, x_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=0.2
            )
            scaler = StandardScaler()
            if self.scale_features:
                x_train = scaler.fit_transform(x_train)
                x_test = scaler.transform(x_test)
                self.scaler = scaler
            else:
                pass
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test

            logger.info("Feature Scaling Completed.")
            return x_train, x_test, y_train, y_test

        except Exception as e:
            logger.error(str(e))
            raise PreprocessingError(
                message="Error Caused While Scaling", details=str(e)
            )

    def dimensionality_reduction(self):
        try:
            if self.df.shape[1] > 250:
                logger.debug(f"{self.df.shape[1]} features found.")
                logger.info("Initiating Dimensionality Reduction")
                pca = PCA(n_components=10)
                self.x_train = pca.fit_transform(self.x_train)
                self.x_test = pca.transform(self.x_test)
                self.pca = pca
            else:
                pass

            logger.info("Dimensionality Reduction Completed.")
            return self.x_train, self.x_test

        except Exception as e:
            logger.error(str(e))
            raise PreprocessingError(
                message="Error Caused While Dimensionality Reduction", details=str(e)
            )

    def process(self):
        logger.info("Starting Preprocessing")
        self.validate()
        self.remove_unwanted_columns()
        self.detect_feature_types()
        self.Problem_detection()
        self.missing_value_handler()
        self.duplicate_handling()
        self.handling_textual_data()
        self.skewness_handling()
        self.handle_outliers()
        self.extract_datetime_features()
        self.encoding()
        self.scaling()
        self.dimensionality_reduction()

        logger.info("Preprocessing Completed.")
        return self.x_train, self.x_test, self.y_train, self.y_test

    def prediction_preprocessor(self, input_df: pd.DataFrame) -> np.ndarray:
        """
        Processes raw input data for inference.
        Ensures columns match training features exactly to prevent Scaler errors.
        """
        logger.info("Processing input data for prediction.")
        try:
            df = input_df.copy()

            # 1. Handling Textual Data (TF-IDF)
            if hasattr(self, "vectorizers") and self.vectorizers:
                for col, vectorizer in self.vectorizers.items():
                    if col in df.columns:
                        # Process text using the global function
                        df[col] = df[col].astype(str).apply(preprocess_text)

                        # Use the fitted vectorizer
                        text_matrix = vectorizer.transform(df[col])

                        # Generate names: Message_tfidf_0, Message_tfidf_1, etc.
                        text_df = pd.DataFrame(
                            text_matrix.toarray(),
                            columns=[
                                f"{col}_tfidf_{i}" for i in range(text_matrix.shape[1])
                            ],
                            index=df.index,
                        )

                        df = df.drop(columns=[col])
                        df = pd.concat([df, text_df], axis=1)

            # 2. Extract Datetime Features
            datetime_cols = self.feature_types.get("datetime", [])
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                    # Note: We must replicate the EXACT logic from extract_datetime_features
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                    df[f"{col}_quarter"] = df[col].dt.quarter
                    df[f"{col}_weekofyear"] = (
                        df[col].dt.isocalendar().week.astype("float")
                    )

                    # Cyclical encoding
                    df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12)
                    df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12)
                    df[f"{col}_dow_sin"] = np.sin(
                        2 * np.pi * df[f"{col}_dayofweek"] / 7
                    )
                    df[f"{col}_dow_cos"] = np.cos(
                        2 * np.pi * df[f"{col}_dayofweek"] / 7
                    )
                    df.drop(columns=[col], inplace=True)

            # 3. Missing Value Imputation
            if self.num_imputer and self.feature_types.get("numerical"):
                num_cols = [
                    c for c in self.feature_types["numerical"] if c in df.columns
                ]
                if num_cols:
                    df[num_cols] = self.num_imputer.transform(df[num_cols])

            if self.cat_imputer and self.feature_types.get("categorical"):
                cat_cols = [
                    c for c in self.feature_types["categorical"] if c in df.columns
                ]
                if cat_cols:
                    df[cat_cols] = self.cat_imputer.transform(df[cat_cols])

            # 4. Encoding
            for col, enc in self.encoders.items():
                if col == "_TARGET_" or col not in df.columns:
                    continue

                # Apply the encoder (Label, OneHot, or Frequency)
                if isinstance(enc, (LabelEncoder, CountFrequencyEncoder)):
                    # Ensure input is 2D for FrequencyEncoder
                    data = (
                        df[[col]] if isinstance(enc, CountFrequencyEncoder) else df[col]
                    )
                    df[col] = enc.transform(data)

            # --- CRITICAL FIX: FEATURE ALIGNMENT ---
            # This ensures the scaler sees the exact same columns as fit time.

            # 1. Add missing columns with 0 (e.g., if a categorical value is missing)
            missing_cols = list(set(self.input_features) - set(df.columns))

            if missing_cols:
                # Create a DataFrame of zeros for all missing columns at once
                # Now columns receives a list, and Pandas is happy.
                missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)

                # Concatenate once to avoid fragmentation
                df = pd.concat([df, missing_df], axis=1)

            # 2. Reorder and De-fragment
            df = df[self.input_features].copy()

            # 5. Scaling
            data_final = df.values
            if self.scaler:
                data_final = self.scaler.transform(df)

            # 6. PCA
            if self.pca:
                data_final = self.pca.transform(data_final)

            return data_final

        except Exception as e:
            logger.error(f"Inference Preprocessing Failed: {str(e)}")
            raise PreprocessingError(
                message="Inference Preprocessing Failed", details=str(e)
            )

        except Exception as e:
            logger.error(f"Inference Preprocessing Failed: {e}")
            raise PreprocessingError(
                message="Inference Preprocessing Failed", details=str(e)
            )


if __name__ == "__main__":
    df = pd.read_excel("temp/branch.xlsx")
    target = "Branch"
    prep = Preprocessor(df, target_column=target)
    x_train, x_test, y_train, y_test = prep.process()
