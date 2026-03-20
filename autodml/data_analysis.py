import pandas as pd
import os
import json
from autodml.utils.logger import get_logger
from autodml.utils.exception import DataAnalysisError
from autodml.utils.utiltiy import Functions
from autodml.preprocessing import Preprocessor


logger = get_logger(__name__)


class DataAnalyzer:
    def __init__(self, df: pd.DataFrame, target: str):
        self.report = {}
        self.df = df
        self.target = target
        self.preprocessor = Preprocessor(df=df, target_column=target)

    def analyze_dataset(self):
        try:
            logger.info("Analyzing dataset overview...")
            df = self.df
            overview = {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_usage_MB": df.memory_usage(deep=True).sum() / 1024**2,
                "duplicate_rows": df.duplicated().sum(),
            }

            self.report["dataset_overview"] = overview

            return overview

        except Exception as e:
            logger.exception("Dataset analysis failed")

            raise DataAnalysisError(
                message="Dataset overview analysis failed", details=str(e)
            )

    def analyze_columns(self):
        try:
            logger.info("Analyzing column statistics...")

            column_info = []

            df = self.df

            for col in df.columns:
                info = {
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "missing_values": df[col].isnull().sum(),
                    "missing_percent": (df[col].isnull().sum() / len(df)) * 100,
                    "unique_values": df[col].nunique(),
                }

                column_info.append(info)

            self.report["column_analysis"] = column_info

            return column_info

        except Exception as e:
            logger.exception("Column analysis failed")

            raise DataAnalysisError(message="Column analysis failed", details=str(e))

    def analyze_target(self):
        try:
            logger.info("Analyzing target variable...")

            if self.target not in self.df.columns:
                raise DataAnalysisError(
                    message="Target column not found",
                    details=f"{self.target} missing in dataset",
                )

            task_type = self.preprocessor.Problem_detection()

            target_stats = {
                "task_type": task_type,
                "unique_values": self.df[self.target].nunique(),
                "missing_values": self.df[self.target].isnull().sum(),
            }

            self.report["target_analysis"] = target_stats

            return target_stats

        except Exception as e:
            logger.exception("Target analysis failed")

            raise DataAnalysisError(message="Target analysis failed", details=str(e))

    def analyze_numeric_features(self):
        try:
            logger.info("Analyzing numeric features...")

            df = self.df

            feature_types = self.preprocessor.detect_feature_types()

            numeric_cols = feature_types["numerical"]

            if len(numeric_cols) == 0:
                logger.warning("No numeric columns found in dataset")
                return {}

            stats = df[numeric_cols].describe().to_dict()

            self.report["numeric_analysis"] = stats

            outlier_report = {}

            for i in numeric_cols:
                self.report["numeric_analysis"][i]["skewness"] = df[i].skew()

                Q1 = df[i].quantile(0.25)
                Q3 = df[i].quantile(0.75)

                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[i] < lower_bound) | (df[i] > upper_bound)]

                outlier_report[i] = {
                    "Q1": Q1,
                    "Q3": Q3,
                    "IQR": IQR,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_count": len(outliers),
                    "outlier_percent": (len(outliers) / len(df)) * 100,
                }

            self.report["outlier_analysis"] = outlier_report

            return self.report

        except Exception as e:
            logger.exception("Numeric analysis failed")

            raise DataAnalysisError(
                message="Numeric feature analysis failed", details=str(e)
            )

    def categorical_feature_analysis(self):
        try:
            logger.info("Analyzing Categorical Features")
            feature_types = self.preprocessor.detect_feature_types()
            categ_cols = feature_types["categorical"]
            categorical_analysis = {}

            df = self.df

            for i in categ_cols:
                if i not in df.columns:
                    logger.warning(f"{i} not found in dataframe. Skipping.")
                    continue

                categorical_analysis[i] = {}
                categorical_analysis[i]["Categories Count"] = df[i].nunique()

            self.report["categorical_analysis"] = categorical_analysis

            logger.info("Analyzed Categorical Columns")

            return categorical_analysis
        except Exception as e:
            logger.error(str(e))
            raise DataAnalysisError(
                message="Error Occured While Analyzing Categorical Data", details=str(e)
            )

    def detect_correlations(self):
        try:
            logger.info("Computing correlations...")

            df = self.df
            feature_types = self.preprocessor.detect_feature_types()
            numeric_cols = feature_types["numerical"]
            numeric_df = df[numeric_cols]

            corr_matrix = numeric_df.corr()

            self.report["correlations"] = corr_matrix

            return corr_matrix

        except Exception as e:
            logger.exception("Correlation analysis failed")

            raise DataAnalysisError(
                message="Correlation detection failed", details=str(e)
            )

    def save_report(self):
        os.makedirs("data/reports/", exist_ok=True)
        with open("data/reports/analysis.json", "w") as f:
            json.dump(self.report, f, indent=4, default=Functions.convert_numpy)

    def generate_report(self):
        logger.info("Generating data analysis report")
        self.analyze_dataset()
        self.analyze_columns()
        self.analyze_target()
        self.analyze_numeric_features()
        self.categorical_feature_analysis()
        self.detect_correlations()
        self.save_report()
        logger.info("Data Analysis Report Generated")
        return self.report


if __name__ == "__main__":
    df = pd.read_csv("temp/HR_Analytics.csv")
    target = "SalarySlab"

    analyzer = DataAnalyzer(df=df, target=target)
    report = analyzer.generate_report()
    print(report)
