import pandas as pd
import numpy as np
from pydantic import create_model
from typing import Optional, Union
from typing import Any
from autodml.preprocessing import Preprocessor, preprocess_text
from autodml.modeling import ModelTrainer
from autodml.evaluation import Evaluator
from autodml.optimization import ModelOptimizer
from autodml.data_analysis import DataAnalyzer
from utils.logger import get_logger
from utils.exception import AutoDMLError
from autodml.registry import ModelRegistry

Models = ModelRegistry()
logger = get_logger(__name__)


class AutoDMLPipeline:
    def __init__(self, target, df):
        self.df = df
        self.target = target
        self.input_features = None
        self.scaler = None
        self.enoders = None
        self.vectorizers = None
        self.input_model = None
        self.problem_type = None
        self.feature_types = None
        self.best_model = None
        self.best_params = None

    def generate_input_model(self):
        fields = {}
        type_map = {
            "numerical": float,
            "categorical": str,
            "text": str,
            "boolean": bool,
            "datetime": str,
            "id": Union[int, str],
        }

        for category, cols in self.feature_types.items():
            if category in ["constant", "all_null"]:
                continue
            python_type = type_map.get(category, Any)

            for col in cols:
                if col == self.target:
                    continue
                fields[col] = (Optional[python_type], None)
        self.input_model = create_model("AutoMLInputModel", **fields)
        return self.input_model

    def prepare_for_prediction(self, input_data: dict):
        validated_data = self.input_model(**input_data)
        input_df = pd.DataFrame([validated_data.model_dump()])
        input_df = input_df.reindex(
            columns=self.original_feature_order, fill_value=np.nan
        )
        return input_df

    def prediction_preprocessor(self, data: pd.DataFrame):
        text_cols = self.feature_types["text"]
        for col in text_cols:
            if col not in df.columns:
                continue

            data[col] = data[col].astype(str).apply(preprocess_text)

            vectorizer = self.vectorizers[col]

            text_matrix = vectorizer.transform(data[col])

            text_df = pd.DataFrame(
                text_matrix.toarray(),
                columns=[f"{col}_tfidf_{i}" for i in range(text_matrix.shape[1])],
            )

            data = data.drop(columns=[col])
            data = pd.concat([data.reset_index(drop=True), text_df], axis=1)

            categorical = self.feature_types["categorical"]
            ids = self.feature_types["id"]

            for i in categorical:
                encoder = self.encoders[i]
                data[i] = encoder.transform[data[i]]
            for i in ids:
                encoder = self.encoders[i]
                data[i] = encoder.transform[data[i]]

            data = self.scaler.transform(data)

            if self.pca is None:
                pass
            else:
                data = self.pca.transform(data)

            return data.values

    def run(self):
        try:
            logger.info("Running AUTODML Pipeline.")
            analyzer = DataAnalyzer(df=self.df, target=self.target)
            analysis = analyzer.generate_report()
            preprocessor = Preprocessor(df=self.df, target_column=self.target)

            self.scaler = preprocessor.scaler_fit
            self.encoders = preprocessor.encoders
            self.vectorizers = preprocessor.vectorizers
            self.input_features = preprocessor.final_feature_names
            self.feature_types = preprocessor.feature_types

            x_train, x_test, y_train, y_test = preprocessor.process()
            problem = preprocessor.problem_type

            self.problem_type = problem

            trainer = ModelTrainer(
                x_train=x_train,
                x_test=x_test,
                y_test=y_test,
                y_train=y_train,
                problem_type=problem,
            )
            model = trainer.get_model()
            optimizer = ModelOptimizer(
                model_name=model, task_type=problem, x_train=x_train, y_train=y_train
            )
            score, param = optimizer.optimize()

            self.best_score = score
            self.best_params = param

            evaluator = Evaluator(
                task_type=problem,
                model=model,
                param=param,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            results = evaluator.evaluate()

            models = Models.get_models(task_type=problem)
            best_model = models[model]
            best_params = optimizer.best_params

            print(results)
            logger.info("AUTODML Pipeline Finished.")
            return best_model, best_params, results, analysis

        except Exception as e:
            logger.error(str(e))
            raise AutoDMLError(
                message="Error Occurred WHile Running AUTODML Pipeline", details=str(e)
            )

    def predict(self, input_data: Union[dict, list, pd.DataFrame]):
        logger.info("Initiating Prediction Process.")

        if isinstance(input_data, dict):
            df_input = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df_input = pd.DataFrame(input_data)
        else:
            df_input = input_data

        input_data = self.generate_input_model()

        processed_df = self.prepare_for_prediction(
            df_input.to_dict(orient="records")[0]
        )

        transformed_features = self.prediction_preprocessor(processed_df)

        final_model = self.best_model_instance.set_params(**self.best_params)
        final_model.fit(self.x_train_ref, self.y_train_ref)

        predictions = final_model.predict(transformed_features)

        return predictions

    def get_input_schema(self):
        """Helper to return the expected Pydantic model for API docs."""
        return self.InputModel.model_json_schema()


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("temp/spam.csv")
    target = "Category"

    autodml = AutoDMLPipeline(df=df, target=target)
    model, parameters, results, analysis = autodml.run()

    inp = {
        "message": "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    }

    print(autodml.predict(inp))
