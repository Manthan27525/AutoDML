import pandas as pd
import numpy as np


class BasicCleaner:
    def smart_remove_unnamed(
        df: pd.DataFrame,
        empty_threshold: float = 0.95,
        info_threshold: float = 0.2,
        aggressive: bool = False,
    ):
        df = df.copy()

        dropped_cols = []
        flagged_cols = []

        n = len(df)

        def is_index_like(series):
            if not pd.api.types.is_numeric_dtype(series):
                return False
            expected = np.arange(n)
            return np.array_equal(series.fillna(-999).values, expected)

        def is_mostly_empty(series):
            return series.isna().mean() >= empty_threshold

        def info_score(series):
            non_null_ratio = 1 - series.isna().mean()
            unique_ratio = series.nunique(dropna=True) / max(len(series.dropna()), 1)
            return 0.7 * non_null_ratio + 0.3 * unique_ratio

        for col in df.columns:
            if "unnamed" in str(col).lower():
                series = df[col]

                index_like = is_index_like(series)
                empty_like = is_mostly_empty(series)
                score = info_score(series)

                # HIGH confidence garbage
                if index_like or empty_like or score < info_threshold:
                    dropped_cols.append(col)

                # MEDIUM suspicious
                else:
                    if aggressive:
                        dropped_cols.append(col)
                    else:
                        flagged_cols.append(col)

        df = df.drop(columns=dropped_cols, errors="ignore")

        return df, dropped_cols


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Salary_Dataset.csv"
    df = pd.read_csv(path)
    df, dropped = BasicCleaner.smart_remove_unnamed(df)
    print(df.head())
    print(dropped)
