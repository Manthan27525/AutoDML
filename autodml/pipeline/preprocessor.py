from autodml.preprocessing.basic_preprocessing import BasicProcessor
from autodml.preprocessing.feature_types import FeatureDetector
from autodml.preprocessing.missing_analysis import MissingAnalyzer
from autodml.feature_engineering.encoding import Encoder
from autodml.preprocessing.missing_handler import MissingValueHandler
from autodml.preprocessing.skewness_analyzer import SkewnessAnalyzer
from autodml.preprocessing.scaling import Scaler

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import pandas as pd


class Preprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.basic_processing = BasicProcessor()
        self.feature_detector = FeatureDetector()
        self.missing_analyzer = MissingAnalyzer()
        self.encoder = Encoder()
        self.missing_handler = MissingValueHandler()
        self.skewness_analyzer = SkewnessAnalyzer()
        self.scaler = Scaler()
        self.pipeline = None
        self.feature_types = None

    def _build_pipeline(self):
        numeric_features = self.feature_types["numeric"]
        categorical_features = self.feature_types["categorical"]

        t1 = ColumnTransformer()

    def fit(self, df: pd.DataFrame, target_column: str):
        self.feature_types = self.feature_detector.auto_detect_feature_types(df)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        self._build_pipeline()

        X_processed = self.pipeline.fit_transform(X)

        return X_processed, y

    def transform(self, df: pd.DataFrame):
        return self.pipeline.transform(df)

    # ---------------------------------------------------
    # FIT + SPLIT
    # ---------------------------------------------------
    def fit_transform(self, df: pd.DataFrame, target_column: str):
        X_processed, y = self.fit(df, target_column)

        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y,
            test_size=self.config.get("test_size", 0.2),
            random_state=self.config.get("random_state", 42),
        )

        return X_train, X_test, y_train, y_test
