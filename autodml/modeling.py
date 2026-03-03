from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Lasso,
    Ridge,
    ElasticNet,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger
from utils.exception import PreprocessingError
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from feature_engine.encoding import CountFrequencyEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score

from sklearn.model_selection import RandomizedSearchCV
from autodml.preprocessing import Preprocessor
import numpy as np
import pandas as pd
import optuna


class ModelTrainer:
    def __init__(
        self,
        problem_type,
        x_train,
        y_train,
        models: dict,
        param_grid: dict,
    ):
        self.models = models
        self.param_grid = param_grid
        self.trained_models = None
        self.best_model = None
        self.best_model_name = None
        self.x_train = x_train
        self.y_train = y_train
        self.problem = problem_type

    def train(self):
        print("Starting Training")
        trained_models = {}

        if self.problem == "Regression":
            for name, model in self.models["Regressors"].items():
                param_grid = self.param_grid["Regressors"][name]

                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=2,
                    cv=5,
                    scoring="r2",
                    n_jobs=-1,
                    random_state=42,
                )

                random_search.fit(self.x_train, self.y_train)

                trained_models[name] = {
                    "Best Params": random_search.best_params_,
                    "Best Score": random_search.best_score_,
                    "Model": random_search.best_estimator_,
                }

        elif self.problem == "Classification":
            for name, model in self.models["Classifiers"].items():
                param_grid = self.param_grid["Classifiers"][name]

                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grid,
                    n_iter=2,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1,
                    random_state=42,
                )

                random_search.fit(self.x_train, self.y_train)

                trained_models[name] = {
                    "Best Params": random_search.best_params_,
                    "Best Score": random_search.best_score_,
                    "Model": random_search.best_estimator_,
                }

        else:
            raise ValueError("Invalid Problem Type")

        self.trained_models = trained_models

        print("Training Completed")

    def best_model_selection(self):
        print("Selecting Best Model")

        best_score = -np.inf
        best_model = None
        best_model_name = None

        for name, result in self.trained_models.items():
            if result["Best Score"] > best_score:
                best_score = result["Best Score"]
                best_model = result
                best_model_name = name

        self.best_model = best_model
        self.best_model_name = best_model_name

        print("Best Model Selected")

    def get_model(self):
        self.train()
        self.best_model_selection()
        return self.best_model, self.best_model_name


if __name__ == "__main__":
    models = {
        "Regressors": {
            "LR": LinearRegression(),
            "L1": Lasso(),
            "L2": Ridge(),
            "EN": ElasticNet(),
            "RF": RandomForestRegressor(),
            "BR": BaggingRegressor(),
            "GB": GradientBoostingRegressor(),
            "KNR": KNeighborsRegressor(),
            "SVR": SVR(),
            "DT": DecisionTreeRegressor(),
        },
        "Classifiers": {
            "LR": LogisticRegression(),
            "RF": RandomForestClassifier(),
            "BC": BaggingClassifier(),
            "GB": GradientBoostingClassifier(),
            "KNC": KNeighborsClassifier(),
            "SVC": SVC(),
            "DT": DecisionTreeClassifier(),
        },
    }
    param_grid = {
        "Regressors": {
            "LR": {
                "fit_intercept": [True, False],
            },
            "L1": {  # Lasso
                "alpha": np.logspace(-5, 1, 30),
                "max_iter": np.arange(1000, 6000, 1000),
            },
            "L2": {  # Ridge
                "alpha": np.logspace(-5, 2, 30),
                "solver": ["auto", "svd", "cholesky", "lsqr"],
            },
            "EN": {  # ElasticNet
                "alpha": np.logspace(-5, 1, 30),
                "l1_ratio": np.linspace(0.0, 1.0, 20),
                "max_iter": np.arange(1000, 6000, 1000),
            },
            "RF": {
                "n_estimators": np.arange(100, 601, 50),
                "max_depth": np.arange(3, 31),
                "min_samples_split": np.arange(2, 21),
                "min_samples_leaf": np.arange(1, 11),
                "max_features": ["sqrt", "log2", None],
            },
            "BR": {
                "n_estimators": np.arange(10, 201, 10),
                "max_samples": np.linspace(0.5, 1.0, 10),
                "max_features": np.linspace(0.5, 1.0, 10),
            },
            "GB": {
                "n_estimators": np.arange(100, 601, 50),
                "learning_rate": np.logspace(-3, -0.5, 30),
                "max_depth": np.arange(3, 11),
                "min_samples_split": np.arange(2, 21),
                "min_samples_leaf": np.arange(1, 11),
                "subsample": np.linspace(0.5, 1.0, 10),
            },
            "KNR": {
                "n_neighbors": np.arange(3, 31),
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            "SVR": {
                "C": np.logspace(-3, 2, 30),
                "epsilon": np.logspace(-4, 0, 30),
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"],
            },
            "DT": {
                "max_depth": np.arange(3, 31),
                "min_samples_split": np.arange(2, 21),
                "min_samples_leaf": np.arange(1, 11),
                "max_features": ["sqrt", "log2", None],
            },
        },
        "Classifiers": {
            "LR": {
                "C": np.logspace(-4, 2, 30),
                "solver": ["lbfgs", "liblinear"],
                "max_iter": np.arange(100, 1001, 100),
            },
            "RF": {
                "n_estimators": np.arange(100, 601, 50),
                "max_depth": np.arange(3, 31),
                "min_samples_split": np.arange(2, 21),
                "min_samples_leaf": np.arange(1, 11),
                "max_features": ["sqrt", "log2", None],
            },
            "BC": {
                "n_estimators": np.arange(10, 201, 10),
                "max_samples": np.linspace(0.5, 1.0, 10),
                "max_features": np.linspace(0.5, 1.0, 10),
            },
            "GB": {
                "n_estimators": np.arange(100, 601, 50),
                "learning_rate": np.logspace(-3, -0.5, 30),
                "max_depth": np.arange(3, 11),
                "min_samples_split": np.arange(2, 21),
                "min_samples_leaf": np.arange(1, 11),
                "subsample": np.linspace(0.5, 1.0, 10),
            },
            "KNC": {
                "n_neighbors": np.arange(3, 31),
                "weights": ["uniform", "distance"],
                "p": [1, 2],
            },
            "SVC": {
                "C": np.logspace(-4, 2, 30),
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"],
            },
            "DT": {
                "max_depth": np.arange(3, 31),
                "min_samples_split": np.arange(2, 21),
                "min_samples_leaf": np.arange(1, 11),
                "max_features": ["sqrt", "log2", None],
            },
        },
    }

    df = pd.read_csv("temp/bmw_sales.csv")
    target = "Model"
    prep = Preprocessor(df=df, target_column=target, scale_features=True)
    x_train, _, y_train, _ = prep.process()
    problem = prep.problem_type
    trainer = ModelTrainer(
        x_train=x_train,
        y_train=y_train,
        problem_type=problem,
        models=models,
        param_grid=param_grid,
    )
    model, name = trainer.get_model()
    print(name)
    print(model["Best Params"])
    print(model["Best Score"])
