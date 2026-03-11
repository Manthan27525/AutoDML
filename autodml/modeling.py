import pandas as pd
import numpy as np
from utils.logger import get_logger
from utils.exception import ModelTrainingError
from sklearn.metrics import r2_score, f1_score
from autodml.registry import ModelRegistry
from autodml.preprocessing import Preprocessor

models = ModelRegistry()

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, problem_type, x_train, x_test, y_train, y_test):
        self.models = models.get_models(problem_type)
        self.model_score = None
        self.best_model = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.problem = problem_type

    def train(self):
        logger.info("Initiating Model Training.")
        model_score = {}

        try:
            if self.problem == "Regression":
                logger.debug("Training Regression Model.")
                for name, model_class in self.models.items():
                    model = model_class()
                    model.fit(self.x_train, self.y_train)
                    pred = model.predict(self.x_test)
                    model_score[name] = r2_score(self.y_test, pred)

            elif self.problem == "Classification":
                logger.debug("Training Classification Model.")
                for name, model_class in self.models.items():
                    model = model_class()
                    model.fit(self.x_train, self.y_train)
                    pred = model.predict(self.x_test)
                    model_score[name] = f1_score(self.y_test, pred, average="weighted")
            else:
                logger.error("Invalid Problem Type")
                raise ValueError("Invalid Problem Type")

            self.model_score = model_score

        except Exception as e:
            logger.error(str(e))
            raise ModelTrainingError(
                message="Error Caused While Model Training", details=str(e)
            )

        logger.info("Model Training Completed.")

    def best_model_selection(self):
        logger.info("Selecting Best Model.")

        best_score = -np.inf
        best_model = None

        for name, result in self.model_score.items():
            if result > best_score:
                best_score = result
                best_model = name

        self.best_model = best_model

        logger.info("Best Model Selected")

    def get_model(self):
        self.train()
        self.best_model_selection()
        return self.best_model


if __name__ == "__main__":
    df = pd.read_csv("temp/bmw_sales.csv")
    target = "Model"
    prep = Preprocessor(df=df, target_column=target, scale_features=True)
    x_train, x_test, y_train, y_test = prep.process()
    problem = prep.problem_type
    trainer = ModelTrainer(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        problem_type=problem,
    )
    model = trainer.get_model()
    print(model)
