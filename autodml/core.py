import pandas as pd
import cloudpickle
import os


class Autodml:
    def __init__(self, target, data):
        self.data = data
        self.target = target
        self.model = None
        self.cat_imputer = None
        self.encoders = {}
        self.feature_types = {}
        self.input_features = None
        self.num_imputer = None
        self.pca = None
        self.preprocess_text = None
        self.scaler = None
        self.skew_transformers = None
        self.vectorizers = None
        self.inputs = None
        self.model = None
        self.analysis_report = None
        self.evaluation_report = None
        self.visualizations_report = None

    def train(self):
        from autodml.pipeline import AutoDMLPipeline

        pipeline = AutoDMLPipeline(target=self.target, df=self.data)
        (
            _,
            _,
            self.evaluation_report,
            self.analysis_report,
            self.visualizations_report,
            meta,
            self.model,
        ) = pipeline.run()

        self.encoders = meta["encoders"]
        self.pca = meta["pca"]
        self.vectorizers = meta["vectorizers"]
        self.skew_transformers = meta["skew_transformers"]
        self.scaler = meta["scaler"]
        self.feature_types = meta["feature_types"]
        self.num_imputer = meta["num_imputer"]
        self.cat_imputer = meta["cat_imputer"]
        self.input_features = meta["input_features"]
        self.preprocess_text = meta["preprocess_text"]
        self.inputs = meta["inputs"]

        return self

    def _preprocess(self, input_df: pd.DataFrame):
        df = input_df.copy()

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

        for col in self.feature_types["datetime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek

                df.drop(columns=[col], inplace=True)

        if self.num_imputer:
            num_cols = self.feature_types["numerical"]
            available_num_cols = [col for col in num_cols if col in df.columns]
            df[available_num_cols] = self.num_imputer.transform(df[available_num_cols])

        if self.cat_imputer:
            cat_cols = self.feature_types["categorical"]
            available_cat_cols = [col for col in cat_cols if col in df.columns]
            df[available_cat_cols] = self.cat_imputer.transform(df[available_cat_cols])

        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
        from feature_engine.encoding import CountFrequencyEncoder

        if self.skew_transformers:
            for col, transformer in self.skew_transformers.items():
                if col in df.columns:
                    df[col] = transformer.transform(df[[col]])

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

        df = df.reindex(columns=self.inputs, fill_value=0)
        missing_cols = list(set(self.inputs) - set(df.columns))
        if missing_cols:
            df = pd.concat(
                [df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1
            )

        df = df[self.inputs]

        if self.scaler:
            df = pd.DataFrame(
                self.scaler.transform(df),
                columns=self.inputs,
                index=df.index,
            )

        if self.pca:
            df = self.pca.transform(df)

        df = df.fillna(0)

        return df

    def predict(self, data):
        df = pd.DataFrame([data])

        X = self._preprocess(df)
        preds = self.model.predict(X)

        if "_TARGET_" in self.encoders:
            preds = self.encoders["_TARGET_"].inverse_transform(preds)

        return preds

    def get_analysis_report(self):
        return self.analysis_report

    def get_evaluation_report(self):
        return self.evaluation_report

    def save(self, path="data/pipeline"):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "pipeline.pkl"), "wb") as f:
            cloudpickle.dump(self, f)

    @classmethod
    def load(self, path="data/pipeline"):
        import cloudpickle
        import os

        with open(os.path.join(path, "pipeline.pkl"), "rb") as f:
            model = cloudpickle.load(f)
        return model
