import pandas as pd
import numpy as np
import dill
import pickle


class AutoDMLPipeline:
    def __init__(
        self,
        model,
        encoders,
        pca,
        vectorizers,
        skew_transformers,
        scaler,
        feature_types,
        num_imputer,
        cat_imputer,
        input_features,
        preprocess_text,
    ):
        self.model = model
        self.encoders = encoders
        self.pca = pca
        self.vectorizers = vectorizers
        self.skew_transformers = skew_transformers
        self.scaler = scaler
        self.feature_types = feature_types
        self.num_imputer = num_imputer
        self.cat_imputer = cat_imputer
        self.input_features = input_features
        self.preprocess_text = preprocess_text

    def _preprocess(self, input_df: pd.DataFrame) -> np.ndarray:
        df = input_df.copy()

        # TEXT
        if self.vectorizers:
            for col, vectorizer in self.vectorizers.items():
                if col in df.columns:
                    df[col] = df[col].astype(str).apply(self.preprocess_text)

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

        # DATETIME
        for col in self.feature_types.get("datetime", []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek

                df.drop(columns=[col], inplace=True)

        # IMPUTATION
        if self.num_imputer:
            num_cols = self.feature_types.get("numerical", [])
            df_num = df.reindex(columns=num_cols, fill_value=np.nan)
            df[num_cols] = self.num_imputer.transform(df_num)

        if self.cat_imputer:
            cat_cols = self.feature_types.get("categorical", [])
            df_cat = df.reindex(columns=cat_cols, fill_value=np.nan)
            df[cat_cols] = self.cat_imputer.transform(df_cat)

        # ENCODING
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
        from feature_engine.encoding import CountFrequencyEncoder

        for col, enc in self.encoders.items():
            if col == "_TARGET_":
                continue

            if isinstance(enc, OrdinalEncoder) and col in df.columns:
                df[col] = enc.transform(df[[col]].astype(str)).ravel()

            elif isinstance(enc, OneHotEncoder) and col in df.columns:
                encoded = enc.transform(df[[col]].astype(str))
                new_cols = [f"{col}_{cat}" for cat in enc.categories_[0]]

                encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
                df = df.drop(columns=[col])
                df = pd.concat([df, encoded_df], axis=1)

            elif isinstance(enc, CountFrequencyEncoder) and col in df.columns:
                df[[col]] = enc.transform(df[[col]].astype(str))

        # ALIGN COLUMNS
        missing_cols = list(set(self.input_features) - set(df.columns))
        if missing_cols:
            df_missing = pd.DataFrame(0, index=df.index, columns=missing_cols)
            df = pd.concat([df, df_missing], axis=1)

        df = df[self.input_features]

        # SKEW
        for col, transformer in self.skew_transformers.items():
            if col in df.columns:
                df[col] = transformer.transform(df[[col]])

        # SCALE
        if self.scaler:
            df = pd.DataFrame(
                self.scaler.transform(df),
                columns=self.input_features,
                index=df.index,
            )

        # PCA
        if self.pca:
            df = self.pca.transform(df)

        return df

    def predict(self, input_df: pd.DataFrame):
        X = self._preprocess(input_df)
        preds = self.model.predict(X)

        # decode target if exists
        if "_TARGET_" in self.encoders:
            preds = self.encoders["_TARGET_"].inverse_transform(preds)

        return preds


model = pickle.load(open("data/model/model.pkl", "rb"))
enc = pickle.load(open("data/preprocessors/encoders.pkl", "rb"))
pca = pickle.load(open("data/preprocessors/pca.pkl", "rb"))
vec = pickle.load(open("data/preprocessors/vectorizers.pkl", "rb"))
skw = pickle.load(open("data/preprocessors/skewtransformers.pkl", "rb"))
sca = pickle.load(open("data/preprocessors/scaler.pkl", "rb"))
ft = pickle.load(open("data/preprocessors/featuretypes.pkl", "rb"))
ni = pickle.load(open("data/preprocessors/numimputer.pkl", "rb"))
ci = pickle.load(open("data/preprocessors/catimputer.pkl", "rb"))
inf = pickle.load(open("data/preprocessors/input.pkl", "rb"))

with open("data/preprocessors/textprocessor.pkl", "rb") as f:
    preprocess_text = pickle.load(f)

pipeline = AutoDMLPipeline(
    model, enc, pca, vec, skw, sca, ft, ni, ci, inf, preprocess_text
)

with open("pipeline.dill", "wb") as f:
    dill.dump(pipeline, f)
