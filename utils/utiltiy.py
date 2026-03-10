import pandas as pd


class Functions:
    @staticmethod
    def safe_read_csv(path):
        try:
            df = pd.read_csv(path)

        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

        except pd.errors.ParserError:
            df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")

        return df
