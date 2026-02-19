import pandas as pd


class MissingAnalyzer:
    @staticmethod
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

    def missing_cols(df):
        a = MissingAnalyzer.analyze(df)
        cols = []
        for i in a.keys():
            if a[i]["missing count"] == 0:
                continue
            else:
                cols.append(i)
        return cols


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    try:
        df = pd.read_csv(path)
        cols = MissingAnalyzer.missing_cols(df)
        print(cols)
    except FileNotFoundError:
        print("File not found. Please check your path.")
