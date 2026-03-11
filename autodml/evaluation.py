import numpy as np
import os
import json

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_percentage_error,
    explained_variance_score,
)

from autodml.preprocessing import Preprocessor
from autodml.optimization import ModelOptimizer
from autodml.modeling import ModelTrainer
from utils.logger import get_logger
from utils.exception import EvaluationError, AutoDMLError
import pandas as pd
from utils.utiltiy import Functions
from autodml.registry import ModelRegistry


models = ModelRegistry()
logger = get_logger(__name__)


class Evaluator:
    def __init__(self, task_type, model, param, x_train, y_train, x_test, y_test):
        self.task_type = task_type
        self.model_name = model
        self.param = param
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.result = {}

    def get_model(self):
        try:
            model_class = models.get_model(
                task_type=self.task_type, model_name=self.model_name
            )

            model = model_class(**self.param)
            model.fit(self.x_train, self.y_train)

            return model
        except Exception as e:
            logger.error(str(e))
            raise AutoDMLError(
                message="Error Caused While Loading Model for Evaluation",
                details=str(e),
            )

    def save_report(self):
        os.makedirs("tests/evaluation", exist_ok=True)
        with open("tests/evaluation/evaluation.json", "w") as f:
            json.dump(self.result, f, indent=4, default=Functions.convert_numpy)

    def evaluate(self):
        logger.info("Initiating Model Evaluation.")
        try:
            model = self.get_model()

            predictions = model.predict(self.x_test)

            if self.task_type == "Regression":
                results = {
                    "MAE": mean_absolute_error(self.y_test, predictions),
                    "MSE": mean_squared_error(self.y_test, predictions),
                    "RMSE": np.sqrt(mean_squared_error(self.y_test, predictions)),
                    "R2": r2_score(self.y_test, predictions),
                    "MAPE": mean_absolute_percentage_error(self.y_test, predictions),
                    "EVS": explained_variance_score(self.y_test, predictions),
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
                    "ROC-AUC": roc_auc_score(self.y_test, predictions),
                    "Confusion Matrix": confusion_matrix(self.y_test, predictions),
                }

            logger.info(f"Evaluation Results: {results}")
            self.result = results

            self.save_report()

            return results

        except Exception as e:
            logger.exception("Model evaluation failed")
            raise EvaluationError(message="Evaluation step failed", details=str(e))


if __name__ == "__main__":
    df = pd.read_csv("temp/olympics.csv")
    target = "medal"
    prep = Preprocessor(df=df, target_column=target, scale_features=True)
    x_train, x_test, y_train, y_test = prep.process()
    problem = prep.problem_type
    trainer = ModelTrainer(
        x_train=x_train,
        x_test=x_test,
        y_test=y_test,
        y_train=y_train,
        problem_type=problem,
    )

    print(prep.feature_types)

    model = trainer.get_model()
    opt = ModelOptimizer(
        model_name=model, task_type=problem, x_train=x_train, y_train=y_train
    )
    score, param = opt.optimize()

    print(score)

    eva = Evaluator(
        task_type=problem,
        model=model,
        param=param,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    results = eva.evaluate()

    print(results)
