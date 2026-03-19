import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from feature_engine.encoding import CountFrequencyEncoder
import pickle

# IMporting All Preprocessors
with open("data/preprocessors/textprocessor.pkl", "rb") as f:
    preprocess_text = pickle.load(f)
with open("data/preprocessors/encoders.pkl", "rb") as f:
    enc = pickle.load(f)
with open("data/preprocessors/pca.pkl", "rb") as f:
    pca = pickle.load(f)
with open("data/preprocessors/scaler.pkl", "rb") as f:
    sca = pickle.load(f)
with open("data/preprocessors/skewtransformers.pkl", "rb") as f:
    skw = pickle.load(f)
with open("data/preprocessors/vectorizers.pkl", "rb") as f:
    vec = pickle.load(f)
with open("data/preprocessors/featuretypes.pkl", "rb") as f:
    ft = pickle.load(f)
with open("data/preprocessors/numimputer.pkl", "rb") as f:
    ni = pickle.load(f)
with open("data/preprocessors/catimputer.pkl", "rb") as f:
    ci = pickle.load(f)
with open("data/preprocessors/input.pkl", "rb") as f:
    inf = pickle.load(f)


class prediction:
    def __init__(self):
        self.encoders = enc
        self.pca = pca
        self.vectorizers = vec
        self.skew_transformers = skw
        self.scaler = sca
        self.feature_types = ft
        self.num_imputer = ni
        self.cat_imputer = ci
        self.input_features = inf

    def prediction_preprocessor(self, input_df: pd.DataFrame) -> np.ndarray:
        df = input_df.copy()
        if hasattr(self, "vectorizers") and self.vectorizers:
            for col, vectorizer in self.vectorizers.items():
                if col in df.columns:
                    df[col] = df[col].astype(str).apply(preprocess_text)

                    text_matrix = vectorizer.transform(df[col])
                    text_df = pd.DataFrame(
                        text_matrix.toarray(),
                        columns=[
                            f"{col}_tfidf_{i}" for i in range(text_matrix.shape[1])
                        ],
                        index=df.index,
                    )

                    df = df.drop(columns=[col])
                    df = pd.concat([df, text_df], axis=1)

        datetime_cols = self.feature_types.get("datetime", [])
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_quarter"] = df[col].dt.quarter
                df[f"{col}_weekofyear"] = df[col].dt.isocalendar().week.astype(float)

                df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12)
                df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12)
                df[f"{col}_dow_sin"] = np.sin(2 * np.pi * df[f"{col}_dayofweek"] / 7)
                df[f"{col}_dow_cos"] = np.cos(2 * np.pi * df[f"{col}_dayofweek"] / 7)

                df.drop(columns=[col], inplace=True)

        if self.num_imputer and self.feature_types.get("numerical"):
            num_cols = [c for c in self.feature_types["numerical"] if c in df.columns]
            if num_cols:
                df[num_cols] = self.num_imputer.transform(df[num_cols])

        if self.cat_imputer and self.feature_types.get("categorical"):
            cat_cols = [c for c in self.feature_types["categorical"] if c in df.columns]
            if cat_cols:
                df[cat_cols] = self.cat_imputer.transform(df[cat_cols])

        for col, enc in self.encoders.items():
            if col == "_TARGET_":
                continue

            if isinstance(enc, OrdinalEncoder):
                if col in df.columns:
                    df[col] = enc.transform(df[[col]].astype(str)).ravel()

            elif isinstance(enc, OneHotEncoder):
                if col in df.columns:
                    encoded = enc.transform(df[[col]].astype(str))

                    new_cols = [f"{col}_{cat}" for cat in enc.categories_[0]]

                    encoded_df = pd.DataFrame(
                        encoded,
                        columns=new_cols,
                        index=df.index,
                    )

                    df = df.drop(columns=[col])
                    df = pd.concat([df, encoded_df], axis=1)

            elif isinstance(enc, CountFrequencyEncoder):
                if col in df.columns:
                    df[[col]] = enc.transform(df[[col]].astype(str))

        missing_cols = list(set(self.input_features) - set(df.columns))
        if missing_cols:
            df_missing = pd.DataFrame(0, index=df.index, columns=missing_cols)
            df = pd.concat([df, df_missing], axis=1)

        extra_cols = list(set(df.columns) - set(self.input_features))
        if extra_cols:
            df = df.drop(columns=extra_cols)

        df = df[self.input_features].copy()

        if hasattr(self, "skew_transformers"):
            for col, transformer in self.skew_transformers.items():
                if col in df.columns:
                    df[col] = transformer.transform(df[[col]])
        if self.scaler:
            df = pd.DataFrame(
                self.scaler.transform(df),
                columns=self.input_features,
                index=df.index,
            )

        if self.pca:
            df = self.pca.transform(df)

        return df.values


pre = prediction()

input_features = {
    """
    # The format for providing input to the model is provided in file input/input.json
    """
    "Message": "As a valued customer, I am pleased to advise you that following recent review of your Mob No. you are awarded with a £1500 Bonus Prize, call 09066364589"
}

with open("data/model/model.pkl", "rb") as f:
    model = pickle.load(f)  # Loading Saved Model
model_input = pre.prediction_preprocessor(
    pd.DataFrame([input_features])
)  # Preparing Model Input

output = enc["_TARGET_"].inverse_transform(
    model.predict(model_input)
)  # Finally Making Predictions From Model

print(output)  # Printing Predictions
