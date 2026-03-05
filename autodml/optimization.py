import optuna
import numpy as np
import pandas as pd

from autodml.preprocessing import Preprocessor
from autodml.modeling import ModelTrainer
from sklearn.model_selection import cross_val_score
from config.models import Models
from config.parameters import Parameters

param_grid = Parameters.get_parameters()
models = Models.get_models()


class ModelOptimizer:
    def __init__(self, model_name, task_type, x_train, y_train, n_trials=30):
        self.task_type = task_type
        self.x_train = x_train
        self.y_train = y_train
        self.n_trials = n_trials
        self.model_name = model_name
        self.best_score = None
        self.best_params = None

    def optimize(self):
        model_class = models[self.task_type][self.model_name]

        def objective(trial):
            params = param_grid[self.task_type][self.model_name](trial)

            model = model_class(**params)

            if self.task_type == "Regression":
                scores = cross_val_score(
                    model, self.x_train, self.y_train, cv=5, scoring="r2", n_jobs=-1
                )
            else:
                scores = cross_val_score(
                    model,
                    self.x_train,
                    self.y_train,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                )

            return scores.mean()

        study = optuna.create_study(direction="maximize")

        study.optimize(objective, n_trials=self.n_trials)

        self.best_score = study.best_value
        self.best_params = study.best_params

        return study.best_value, study.best_params


if __name__ == "__main__":
    df = pd.read_csv("temp/mushrooms.csv")
    target = "class"
    prep = Preprocessor(df=df, target_column=target, scale_features=True)
    x_train, x_test, y_train, y_test = prep.process()
    problem = prep.problem_type
    trainer = ModelTrainer(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        problem_type=problem,
    )
    model = trainer.get_model()
    opt = ModelOptimizer(
        model_name=model, task_type=problem, x_train=x_train, y_train=y_train
    )
    score, param = opt.optimize()

    print(score)
    print(param)
