import pandas as pd


class Validators:
    def validate_not_empty(df):
        return df.empty

    def validate_target_exists(df, target):
        if target in df.columns:
            return True
        else:
            return False

    def validate_no_duplicate_columns(df):
        for i in df.columns.duplicated():
            if i == True:
                return False
            else:
                return True


if __name__ == "__main__":
    path = r"D:\Project\AutoDML\temp\Student_Performance.csv"
    df = pd.read_csv(path)
    target = "Peformance Index"
    print(Validators.validate_not_empty(df))
    print(Validators.validate_target_exists(df, target))
    print(Validators.validate_no_duplicate_columns(df))
