import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from autodml.utils.logger import get_logger
from autodml.utils.exception import DataVisualizationError
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import streamlit as st
import os

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")


class DataVisualizer:
    def __init__(self, model, feature_names, df: pd.DataFrame, target):
        self.df = df
        self.target = target
        self.model = model
        self.feature_names = feature_names

    def clean_for_visualization(self):
        df = self.df.copy()

        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

        df = df.drop_duplicates()

        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        df[col] = parsed
                except Exception as e:
                    raise DataVisualizationError(
                        message="Error Occurred While Data Visualization",
                        details=str(e),
                    )
        self.df = df
        return df

    def plot_missing_values(self):
        try:
            df = self.df.copy()

            missing = df.isnull().sum()
            missing = missing[missing > 0]

            if missing.empty:
                logger.info("No missing values found")
                return None

            missing_percent = (missing / len(df)) * 100

            plot_df = pd.DataFrame(
                {"Missing Count": missing, "Missing %": missing_percent}
            ).sort_values(by="Missing Count", ascending=False)

            plt.figure(figsize=(10, 5))

            plot_df["Missing Count"].plot(kind="bar")

            plt.title("Missing Values Analysis")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()

            return plt.gcf()

        except Exception as e:
            raise DataVisualizationError(
                message="Error Occurred While Plotting Missing Values",
                details=str(e),
            )

    def plot_numerical_distributions(self, max_cols=6):
        try:
            df = self.df.copy()

            num_cols = df.select_dtypes(include=np.number).columns.tolist()

            if not num_cols:
                logger.info("No numerical columns found")
                return None

            num_cols = num_cols[:max_cols]

            figs = []

            for col in num_cols:
                data = df[col].dropna()

                if data.empty:
                    continue

                plt.figure(figsize=(6, 4))

                sns.histplot(data, kde=True)

                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.tight_layout()

                figs.append(plt.gcf())

            return figs if figs else None

        except Exception as e:
            logger.error(str(e))
            raise DataVisualizationError(
                message="Error in numerical distribution plots", details=str(e)
            )

    def plot_boxplots(self, max_cols=6):
        try:
            df = self.df.copy()

            num_cols = df.select_dtypes(include=np.number).columns.tolist()

            if not num_cols:
                logger.info("No numerical columns found for boxplots")
                return None

            num_cols = num_cols[:max_cols]

            figs = []

            for col in num_cols:
                data = df[col].dropna()

                if data.empty:
                    continue

                plt.figure(figsize=(6, 4))

                sns.boxplot(x=data)

                plt.title(f"Boxplot of {col}")
                plt.xlabel(col)
                plt.tight_layout()

                figs.append(plt.gcf())

            return figs if figs else None

        except Exception as e:
            logger.error(str(e))
            raise DataVisualizationError(
                message="Error in boxplot generation", details=str(e)
            )

    def plot_correlation_heatmap(self, max_cols=20):
        try:
            df = self.df.copy()

            num_df = df.select_dtypes(include=np.number)

            if num_df.shape[1] < 2:
                logger.info("Not enough numerical features for correlation heatmap")
                return None

            if num_df.shape[1] > max_cols:
                num_df = num_df.iloc[:, :max_cols]

            corr = num_df.corr()

            plt.figure(figsize=(10, 6))

            sns.heatmap(corr, cmap="coolwarm", annot=False, linewidths=0.5)

            plt.title("Feature Correlation Heatmap")
            plt.tight_layout()

            return plt.gcf()

        except Exception as e:
            logger.error(str(e))
            raise DataVisualizationError(
                message="Error in correlation heatmap", details=str(e)
            )

    def plot_target_distribution(self):
        try:
            df = self.df.copy()

            if self.target is None or self.target not in df.columns:
                logger.info("Target column not provided")
                return None

            target_series = df[self.target].dropna()

            if target_series.empty:
                logger.info("Target column is empty")
                return None

            plt.figure(figsize=(6, 4))

            if target_series.dtype == "object" or target_series.nunique() < 20:
                sns.countplot(x=target_series)
                plt.xticks(rotation=45)
                plt.title(f"Target Distribution (Classification): {self.target}")
            else:
                sns.histplot(target_series, kde=True)
                plt.title(f"Target Distribution (Regression): {self.target}")

            plt.tight_layout()

            return plt.gcf()

        except Exception as e:
            logger.error(str(e))
            raise DataVisualizationError(
                message="Error in target distribution plot", details=str(e)
            )

    def plot_feature_vs_target(self, max_cols=5):
        try:
            df = self.df.copy()

            if self.target is None or self.target not in df.columns:
                logger.info("Target column not provided")
                return None

            target = df[self.target]
            figs = []

            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if self.target in num_cols:
                num_cols.remove(self.target)
            if self.target in cat_cols:
                cat_cols.remove(self.target)

            is_classification = target.dtype == "object" or target.nunique() < 20

            for col in num_cols[:max_cols]:
                data = df[[col, self.target]].dropna()

                if data.empty:
                    continue

                plt.figure(figsize=(6, 4))

                if is_classification:
                    sns.boxplot(x=data[self.target], y=data[col])
                    plt.title(f"{col} vs {self.target} (Class Distribution)")
                else:
                    sns.scatterplot(x=data[col], y=data[self.target])
                    plt.title(f"{col} vs {self.target} (Regression)")

                plt.tight_layout()
                figs.append(plt.gcf())

            for col in cat_cols[:max_cols]:
                data = df[[col, self.target]].dropna()

                if data.empty:
                    continue

                top_categories = data[col].value_counts().index[:10]
                data = data[data[col].isin(top_categories)]

                plt.figure(figsize=(6, 4))

                if is_classification:
                    sns.countplot(x=col, hue=self.target, data=data)
                    plt.title(f"{col} vs {self.target}")
                    plt.xticks(rotation=45)
                else:
                    sns.boxplot(x=col, y=self.target, data=data)
                    plt.title(f"{col} vs {self.target}")
                    plt.xticks(rotation=45)

                plt.tight_layout()
                figs.append(plt.gcf())

            return figs if figs else None

        except Exception as e:
            logger.error(str(e))
            raise DataVisualizationError(
                message="Error in feature vs target plots", details=str(e)
            )

    def plot_categorical_distributions(self, max_cols=5, top_n=10):
        try:
            df = self.df.copy()

            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if not cat_cols:
                logger.info("No categorical columns found")
                return None

            cat_cols = cat_cols[:max_cols]

            figs = []

            for col in cat_cols:
                data = df[col].dropna()

                if data.empty:
                    continue

                top_categories = data.value_counts().head(top_n)

                plt.figure(figsize=(6, 4))

                top_categories.plot(kind="bar")

                plt.title(f"Top {top_n} Categories in {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=45)
                plt.tight_layout()

                figs.append(plt.gcf())

            return figs if figs else None

        except Exception as e:
            logger.error(str(e))
            raise DataVisualizationError(
                message="Error in categorical distribution plots", details=str(e)
            )

    def plot_feature_importance(self, top_n=15):
        try:
            model = self.model

            if model is None:
                logger.info("Model not provided")
                return None

            importances = None

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
                if isinstance(importances, np.ndarray) and importances.ndim > 1:
                    importances = importances.flatten()

            if importances is None:
                logger.info("Model does not support feature importance")
                return None

            feature_names = self.feature_names

            if feature_names is None or len(feature_names) != len(importances):
                feature_names = [f"Feature_{i}" for i in range(len(importances))]

            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})

            imp_df = imp_df.sort_values(by="importance", ascending=False).head(top_n)

            plt.figure(figsize=(8, 5))

            sns.barplot(x="importance", y="feature", data=imp_df)

            plt.title("Top Feature Importance")
            plt.tight_layout()

            return plt.gcf()

        except Exception as e:
            logger.error(str(e))
            raise DataVisualizationError(
                message="Error in feature importance plot", details=str(e)
            )

    def generate_all_visuals(self):
        return {
            "missing": self.plot_missing_values(),
            "distributions": self.plot_numerical_distributions(),
            "boxplots": self.plot_boxplots(),
            "correlation": self.plot_correlation_heatmap(),
            "target": self.plot_target_distribution(),
            "feature_vs_target": self.plot_feature_vs_target(),
            "categorical": self.plot_categorical_distributions(),
            "feature_importance": self.plot_feature_importance(),
        }

    def save_plots(self, plots: dict, folder="data/plots"):
        os.makedirs(folder, exist_ok=True)

        for name, plot in plots.items():
            if plot is None:
                continue

            if isinstance(plot, list):
                for i, fig in enumerate(plot):
                    fig.savefig(f"{folder}/{name}_{i}.png")
            else:
                plot.savefig(f"{folder}/{name}.png")

    def generate_html_report(self, plots, output_file="data/reports/report.html"):
        self.save_plots(plots)
        os.makedirs("data/reports", exist_ok=True)

        html = "<html><head><title>AutoDML Report</title></head><body>"
        html += "<h1>AutoDML Analysis Report</h1>"

        for name, plot in plots.items():
            if plot is None:
                continue

            html += f"<h2>{name.replace('_', ' ').title()}</h2>"

            if isinstance(plot, list):
                for i, fig in enumerate(plot):
                    path = f"data/plots/{name}_{i}.png"
                    fig.savefig(path)
                    html += f"<img src='{path}' width='600'><br>"
            else:
                path = f"data/plots/{name}.png"
                plot.savefig(path)
                html += f"<img src='{path}' width='600'><br>"

        html += "</body></html>"

        with open(output_file, "w") as f:
            f.write(html)

        return output_file

    def generate_pdf_report(self, plots, output_file="data/reports/report.pdf"):
        os.makedirs("data/reports", exist_ok=True)

        doc = SimpleDocTemplate(output_file)
        styles = getSampleStyleSheet()

        elements = []

        elements.append(Paragraph("AutoDML Report", styles["Title"]))
        elements.append(Spacer(1, 20))

        for name, plot in plots.items():
            if plot is None:
                continue

            elements.append(
                Paragraph(name.replace("_", " ").title(), styles["Heading2"])
            )

            if isinstance(plot, list):
                for i, fig in enumerate(plot):
                    path = f"data/plots/{name}_{i}.png"
                    fig.savefig(path)
                    elements.append(Image(path, width=400, height=250))
            else:
                path = f"data/plots/{name}.png"
                plot.savefig(path)
                elements.append(Image(path, width=400, height=250))

            elements.append(Spacer(1, 20))

        doc.build(elements)

        return output_file


if __name__ == "__main__":
    df = pd.read_csv("temp/bmw_sales.csv")
    visualizer = DataVisualizer(df=df, target="Avg_Price_EUR")
    plots = visualizer.generate_all()

    st.title("AutoDML Dashboard")

    for key, value in plots.items():
        st.subheader(key)

        if isinstance(value, list):
            for fig in value:
                st.pyplot(fig)
        elif value:
            st.pyplot(value)
