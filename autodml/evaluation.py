import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from autodml.preprocessing import Preprocessor
from autodml.modeling import ModelTrainer
from utils.logger import get_logger
from utils.exception import AutoDMLError
import pandas as pd

from config.models import Models

logger = get_logger(__name__)


class Evaluator:
    def __init__(self, task_type, model, name, x_test, y_test):
        self.task_type = task_type
        self.config = model
        self.model_name = name
        self.x_test = x_test
        self.y_test = y_test

    def get_model(self):
        lt = Models.get_models()
        model = lt[self.task_type][self.model_name]

        model = self.config["Model"]

        return model

    def evaluate(self):
        try:
            logger.info("Starting model evaluation...")

            model = self.get_model()

            predictions = model.predict(self.x_test)

            if self.task_type == "Regression":
                results = {
                    "MAE": mean_absolute_error(self.y_test, predictions),
                    "MSE": mean_squared_error(self.y_test, predictions),
                    "RMSE": np.sqrt(mean_squared_error(self.y_test, predictions)),
                    "R2": r2_score(self.y_test, predictions),
                }

            else:
                results = {
                    "Accuracy": accuracy_score(self.y_test, predictions),
                    "Precision": precision_score(
                        self.y_test, predictions, average="weighted"
                    ),
                    "Recall": recall_score(
                        self.y_test, predictions, average="weighted"
                    ),
                    "F1": f1_score(self.y_test, predictions, average="weighted"),
                }

            logger.info(f"Evaluation Results: {results}")

            return results

        except Exception as e:
            logger.exception("Model evaluation failed")

            raise AutoDMLError(message="Evaluation step failed", details=str(e))


if __name__ == "__main__":
    df = pd.read_csv("temp/coke.csv")
    target = "Close"
    prep = Preprocessor(df=df, target_column=target, scale_features=True)
    x_train, x_test, y_train, y_test = prep.process()
    problem = prep.problem_type
    trainer = ModelTrainer(x_train=x_train, y_train=y_train, problem_type=problem)
    model, name = trainer.get_model()
    eva = Evaluator(
        task_type=problem, model=model, name=name, x_test=x_test, y_test=y_test
    )
    results = eva.evaluate()

    print(results)
