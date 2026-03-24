import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from autodml.utils.logger import get_logger
from autodml.utils.exception import DataVisualizationError

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.1)


def safe_plot(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise DataVisualizationError(
                message=f"Error Occurred in {func.__name__}", details=str(e)
            )
            return None

    return wrapper


class DataVisualizer:
    def __init__(self, model=None, feature_names=None, df=None, target=None):
        self.df = df
        self.target = target
        self.model = model
        self.feature_names = feature_names

    def clean_for_visualization(self):
        try:
            df = self.df.copy()

            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            df = df.drop_duplicates()

            for col in df.columns:
                if df[col].dtype == "object":
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        df[col] = parsed
        except Exception as e:
            logger.error(str(e))
            raise DataVisualizationError(
                message="Error occured while Data Visualization", details=str(e)
            )

        self.df = df
        return df

    def detect_dataset_type(self):
        df = self.df

        text_cols = [
            col
            for col in df.columns
            if df[col].dtype == "object" and df[col].astype(str).str.len().mean() > 20
        ]

        if text_cols and len(df.columns) <= 3:
            return "text"

        return "structured"

    @safe_plot
    def plot_missing_values(self):
        df = self.df
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            return None

        plt.figure(figsize=(8, 4))
        missing.sort_values().plot(
            kind="bar", color=sns.color_palette("magma", len(missing))
        )

        plt.title("Missing Values", fontsize=14, fontweight="bold")
        plt.tight_layout(pad=1.2)
        return plt.gcf()

    @safe_plot
    def plot_numerical_distributions(self, max_cols=6):
        df = self.df
        num_cols = df.select_dtypes(include=np.number).columns[:max_cols]

        if len(num_cols) == 0:
            return None

        figs = []

        for col in num_cols:
            data = df[col].dropna()
            if data.empty:
                continue

            plt.figure(figsize=(6, 4))
            sns.histplot(data, kde=True, bins=30, color=sns.color_palette()[0])

            plt.title(col, fontsize=13, fontweight="bold")
            plt.grid(alpha=0.3)
            plt.tight_layout(pad=1.2)

            figs.append(plt.gcf())

        return figs if figs else None

    @safe_plot
    def plot_boxplots(self, max_cols=6):
        df = self.df
        num_cols = df.select_dtypes(include=np.number).columns[:max_cols]

        if len(num_cols) == 0:
            return None

        figs = []

        for col in num_cols:
            data = df[col].dropna()
            if data.empty:
                continue

            plt.figure(figsize=(6, 4))
            sns.boxplot(x=data, color=sns.color_palette()[1])

            plt.title(col, fontsize=13, fontweight="bold")
            plt.tight_layout(pad=1.2)

            figs.append(plt.gcf())

        return figs if figs else None

    @safe_plot
    def plot_categorical_distributions(self, max_cols=5):
        df = self.df
        cat_cols = df.select_dtypes(include=["object", "category"]).columns[:max_cols]

        if len(cat_cols) == 0:
            return None

        figs = []

        for col in cat_cols:
            data = df[col].dropna()
            if data.empty:
                continue

            plt.figure(figsize=(6, 4))
            sns.countplot(x=data, palette="viridis")

            plt.xticks(rotation=45)
            plt.title(col, fontsize=13, fontweight="bold")
            plt.tight_layout(pad=1.2)

            figs.append(plt.gcf())

        return figs if figs else None

    @safe_plot
    def plot_categorical_vs_categorical(self):
        df = self.df
        cat_cols = df.select_dtypes(include=["object"]).columns

        if len(cat_cols) < 2:
            return None

        figs = []

        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                data = df[[cat_cols[i], cat_cols[j]]].dropna()

                if data.empty:
                    continue

                cross = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])

                if cross.empty:
                    continue

                plt.figure(figsize=(6, 4))
                sns.heatmap(
                    cross,
                    annot=True,
                    fmt="d",
                    cmap="coolwarm",
                    linewidths=0.5,
                    linecolor="white",
                )

                plt.title(f"{cat_cols[i]} vs {cat_cols[j]}", fontweight="bold")
                plt.tight_layout(pad=1.2)

                figs.append(plt.gcf())

        return figs if figs else None

    @safe_plot
    def plot_numerical_vs_numerical(self):
        df = self.df
        num_cols = df.select_dtypes(include=np.number).columns

        if len(num_cols) < 2:
            return None

        figs = []

        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                data = df[[num_cols[i], num_cols[j]]].dropna()

                if data.empty:
                    continue

                plt.figure(figsize=(6, 4))
                sns.scatterplot(
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    alpha=0.7,
                    color=sns.color_palette()[2],
                )

                plt.title(f"{num_cols[i]} vs {num_cols[j]}", fontweight="bold")
                plt.tight_layout(pad=1.2)

                figs.append(plt.gcf())

        return figs if figs else None

    @safe_plot
    def plot_numerical_vs_categorical(self):
        df = self.df
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include=["object"]).columns

        if len(num_cols) == 0 or len(cat_cols) == 0:
            return None

        figs = []

        for num in num_cols:
            for cat in cat_cols:
                data = df[[num, cat]].dropna()

                if data.empty:
                    continue

                plt.figure(figsize=(6, 4))
                sns.boxplot(x=cat, y=num, data=data, palette="Set2")

                plt.xticks(rotation=45)
                plt.title(f"{num} vs {cat}", fontweight="bold")
                plt.tight_layout(pad=1.2)

                figs.append(plt.gcf())

        return figs if figs else None

    @safe_plot
    def plot_target_distribution(self):
        df = self.df

        if not self.target or self.target not in df.columns:
            return None

        data = df[self.target].dropna()

        if data.empty:
            return None

        plt.figure(figsize=(6, 4))

        if data.dtype == "object" or data.nunique() < 20:
            sns.countplot(x=data, palette="viridis")
            plt.xticks(rotation=45)
        else:
            sns.histplot(data, kde=True, color=sns.color_palette()[3])

        plt.title(f"Target: {self.target}", fontweight="bold")
        plt.tight_layout(pad=1.2)

        return plt.gcf()

    @safe_plot
    def plot_wordcloud(self):
        df = self.df

        text_cols = [
            col
            for col in df.columns
            if df[col].dtype == "object" and df[col].astype(str).str.len().mean() > 20
        ]

        if not text_cols:
            return None

        text = " ".join(df[text_cols[0]].dropna().astype(str))

        if not text.strip():
            return None

        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            colormap="viridis",
        ).generate(text)

        plt.figure(figsize=(8, 4))
        plt.imshow(wc)
        plt.axis("off")
        plt.tight_layout(pad=1.2)

        return plt.gcf()

    @safe_plot
    def plot_feature_importance(self):
        if self.model is None:
            return None

        importances = None

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_

        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_)

            if isinstance(importances, np.ndarray) and importances.ndim > 1:
                importances = importances.flatten()

        if importances is None:
            return None

        feature_names = self.feature_names

        if feature_names is None or len(feature_names) != len(importances):
            logger.warning("Feature names mismatch — auto generating names")
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})

        df_imp = df_imp.sort_values(by="importance", ascending=False).head(15)

        plt.figure(figsize=(6, 4))
        sns.barplot(
            x="importance",
            y="feature",
            data=df_imp,
            hue="feature",
            palette="magma",
            legend=False,
        )

        plt.title("Feature Importance", fontweight="bold")
        plt.tight_layout()

        fig = plt.gcf()
        plt.close()

        return fig

    def generate_all_visuals(self):
        self.clean_for_visualization()
        mode = self.detect_dataset_type()

        plots = {}

        if mode == "text":
            logger.info("Text dataset detected")
            plots["wordcloud"] = self.plot_wordcloud()
            plots["target"] = self.plot_target_distribution()
            return plots

        logger.info("Structured dataset detected")

        plots["missing"] = self.plot_missing_values()
        plots["num_dist"] = self.plot_numerical_distributions()
        plots["box"] = self.plot_boxplots()
        plots["cat"] = self.plot_categorical_distributions()
        plots["target"] = self.plot_target_distribution()
        plots["cat_vs_cat"] = self.plot_categorical_vs_categorical()
        plots["num_vs_cat"] = self.plot_numerical_vs_categorical()
        plots["num_vs_num"] = self.plot_numerical_vs_numerical()
        plots["feature_imp"] = self.plot_feature_importance()

        return plots

    def save_plots(self, plots):
        os.makedirs("data/plots", exist_ok=True)

        for name, plot in plots.items():
            if not plot:
                continue

            try:
                if isinstance(plot, list):
                    for i, fig in enumerate(plot):
                        fig.savefig(f"data/plots/{name}_{i}.png")
                        plt.close(fig)
                else:
                    plot.savefig(f"data/plots/{name}.png")
                    plt.close(plot)
            except Exception as e:
                logger.error(f"Failed saving {name}: {str(e)}")

    def generate_pdf_report(self, plots, output_file="data/reports/report.pdf"):
        try:
            os.makedirs("data/plots", exist_ok=True)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            doc = SimpleDocTemplate(output_file, pagesize=A4)
            styles = getSampleStyleSheet()

            elements = []

            elements.append(Paragraph("AutoDML Report", styles["Title"]))
            elements.append(Spacer(1, 20))

            try:
                elements.append(
                    Paragraph(f"Dataset Shape: {self.df.shape}", styles["Normal"])
                )
                elements.append(Spacer(1, 15))
            except Exception as e:
                raise DataVisualizationError(
                    message="Error Occurred While Generating Report", details=str(e)
                )
            for name, plot in plots.items():
                if not plot:
                    continue

                elements.append(
                    Paragraph(name.replace("_", " ").title(), styles["Heading2"])
                )
                elements.append(Spacer(1, 10))

                try:
                    if isinstance(plot, list):
                        for i, fig in enumerate(plot):
                            if fig is None:
                                continue

                            img_path = f"data/plots/{name}_{i}.png"

                            fig.savefig(img_path)
                            plt.close(fig)

                            elements.append(Image(img_path, width=450, height=250))
                            elements.append(Spacer(1, 15))

                    else:
                        img_path = f"data/plots/{name}.png"

                        plot.savefig(img_path)
                        plt.close(plot)

                        elements.append(Image(img_path, width=450, height=250))
                        elements.append(Spacer(1, 15))

                except Exception as e:
                    logger.error(f"Error adding plot {name}: {str(e)}")
                    continue

            doc.build(elements)

            logger.info(f"PDF report generated at {output_file}")
            return output_file

        except Exception as e:
            raise DataVisualizationError(
                message="Error generating PDF report", details=str(e)
            )
