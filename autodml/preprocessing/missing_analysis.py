import pandas as pd


class MissingAnalyzer:
    @staticmethod  # Added decorator since 'self' isn't used
    def analyze(df):
        analysis = {}
        total_rows = len(df)

        for i in df.columns:
            missing_count = df[i].isna().sum()

            analysis[i] = {
                "missing count": missing_count,
                "missing percentage": f"{(missing_count / total_rows) * 100:.2f}%",
            }
        return analysis


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    try:
        df = pd.read_csv(path)
        analysis = MissingAnalyzer.analyze(df)
        print(analysis)
    except FileNotFoundError:
        print("File not found. Please check your path.")
