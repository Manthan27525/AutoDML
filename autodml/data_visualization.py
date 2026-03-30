import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from reportlab.platypus import SimpleDocTemplate, Image, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import matplotlib
from autodml.utils.logger import get_logger
from autodml.utils.exception import DataVisualizationError
import gc

matplotlib.use("Agg")

logger = get_logger(__name__)

sns.set_theme(style="whitegrid")
sns.set_palette("husl")
sns.set_context("notebook", font_scale=1.1)


class DataVisualizer:
    def __init__(self, model, feature_names, df, target, feature_types):
        self.df = df
        self.target = target
        self.model = model
        self.feature_names = feature_names
        self.feature_types = feature_types

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

    def plot_numerical_distributions(self, max_cols=5):
        try:
            df = self.df
            num_cols = self.feature_types["numerical"][:max_cols]

            figs = []

            for col in num_cols:
                data = df[col].dropna()

                if data.empty:
                    continue

                if len(data) > 5000:
                    data = data.sample(5000)

                plt.figure(figsize=(4, 3))
                sns.histplot(data, kde=True)

                plt.title(col)
                fig = plt.gcf()
                figs.append(fig)
                plt.close(fig)

            gc.collect()
            return figs if figs else None
        except Exception as e:
            raise DataVisualizationError(
                message="Error While Plotting Numerical Distributions", details=str(e)
            )

    def plot_boxplots(self, max_cols=6):
        try:
            df = self.df
            num_cols = self.feature_types["numerical"][:max_cols]

            if len(num_cols) == 0:
                return None

            figs = []

            for col in num_cols:
                data = df[col].dropna()
                if data.empty:
                    continue

                plt.figure(figsize=(6, 4))
                sns.boxplot(x=data)

                plt.title(col, fontsize=13, fontweight="bold")
                plt.tight_layout(pad=1.2)

                figs.append(plt.gcf())
            gc.collect()
            return figs if figs else None
        except Exception as e:
            raise DataVisualizationError(
                message="Error While Plotting Boxplots", details=str(e)
            )

    def plot_categorical_distributions(self, max_cols=5):
        try:
            df = self.df
            cat_cols = self.feature_types["categorical"][:max_cols]

            figs = []

            for col in cat_cols:
                data = df[col].dropna()

                if data.empty:
                    continue

                if len(data) > 5000:
                    data = data.sample(5000)

                top = data.value_counts().nlargest(10).index
                data = data[data.isin(top)]

                plt.figure(figsize=(4, 3))
                sns.countplot(x=data)

                plt.xticks(rotation=45)
                fig = plt.gcf()
                figs.append(fig)
                plt.close(fig)
            gc.collect()
            return figs if figs else None
        except Exception as e:
            raise DataVisualizationError(
                message="Error Occured WHile Plotting Categorical Distributions",
                details=str(e),
            )

    def plot_numerical_vs_numerical(self):
        try:
            df = self.df
            num_cols = self.feature_types["numerical"][:4]

            figs = []

            pairs = [
                (num_cols[i], num_cols[j])
                for i in range(len(num_cols))
                for j in range(i + 1, len(num_cols))
            ]

            pairs = pairs[:3]

            for x, y in pairs:
                data = df[[x, y]].dropna()

                if len(data) > 3000:
                    data = data.sample(3000)

                plt.figure(figsize=(4, 3))
                sns.scatterplot(x=data[x], y=data[y])

                plt.title(f"{x} vs {y}")
                fig = plt.gcf()
                figs.append(fig)
                plt.close(fig)

            gc.collect()
            return figs if figs else None

        except Exception as e:
            raise DataVisualizationError(
                message="Error While Plotting Numerical vs Numerical Plots",
                details=str(e),
            )

    def plot_target_distribution(self):
        try:
            df = self.df
            data = df[self.target].dropna()

            if len(data) > 5000:
                data = data.sample(5000)

            plt.figure(figsize=(4, 3))

            if data.nunique() < 20:
                sns.countplot(x=data)
            else:
                sns.histplot(data, kde=True)

            fig = plt.gcf()
            plt.close(fig)
            gc.collect()
            return fig
        except Exception as e:
            raise DataVisualizationError(
                message="Error While Plotting Target Distribution", details=str(e)
            )

    def plot_wordcloud(self):
        try:
            df = self.df

            text_cols = self.feature_types["text"]

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
            plt.tight_layout()
            gc.collect()
            return plt.gcf()
        except Exception as e:
            raise DataVisualizationError(
                message="Error While Plotting Wordcloud", details=str(e)
            )

    def generate_all_visuals(self):
        self.clean_for_visualization()

        plots = {}

        plots["num_dist"] = self.plot_numerical_distributions()
        plots["box"] = self.plot_boxplots()
        plots["cat"] = self.plot_categorical_distributions()
        plots["target"] = self.plot_target_distribution()
        plots["num_vs_num"] = self.plot_numerical_vs_numerical()
        plots["word_cloud"] = self.plot_wordcloud()

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
            os.makedirs("data/reports", exist_ok=True)

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
