import pandas as pd

import os
import pandas as pd


class DataImporter:
    SUPPORTED_FORMATS = [".csv", ".xlsx", ".xls", ".json"]

    @staticmethod
    def load(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found at path: {path}")
        file_extension = os.path.splitext(path)[1].lower()

        if file_extension not in DataImporter.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format '{file_extension}'. "
                f"Supported formats: {DataImporter.SUPPORTED_FORMATS}"
            )

        try:
            if file_extension == ".csv":
                df = pd.read_csv(path)

            elif file_extension in [".xlsx", ".xls"]:
                df = pd.read_excel(path)

            elif file_extension == ".json":
                df = pd.read_json(path)
            else:
                raise ValueError("Unsupported file format.")

        except Exception as e:
            raise RuntimeError(f"Error reading file: {e}")

        if df.empty:
            raise ValueError("Loaded file is empty.")

        return df


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    df = DataImporter.load(path)
    print(df.head())
