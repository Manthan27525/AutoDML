import numpy as np
import pandas as pd


class Functions:
    @staticmethod
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()

        if isinstance(obj, pd.Series):
            return obj.to_dict()

        raise TypeError(f"Type {type(obj)} not serializable")
