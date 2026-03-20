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
from autodml.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    def __init__(self):
        self.regression = {
            "LR": LinearRegression,
            "L1": Lasso,
            "L2": Ridge,
            "EN": ElasticNet,
            "RF": RandomForestRegressor,
            "BR": BaggingRegressor,
            "GB": GradientBoostingRegressor,
            "KNR": KNeighborsRegressor,
            "SVR": SVR,
            "DT": DecisionTreeRegressor,
        }

        self.classification = {
            "LR": LogisticRegression,
            "RF": RandomForestClassifier,
            "BC": BaggingClassifier,
            "GB": GradientBoostingClassifier,
            "KNC": KNeighborsClassifier,
            "SVC": SVC,
            "DT": DecisionTreeClassifier,
        }

    def get_models(self, task_type):
        if task_type == "Regression":
            return self.regression

        elif task_type == "Classification":
            return self.classification

        else:
            raise ValueError("Invalid task type")

    def get_model(self, model_name, task_type):
        models = self.get_models(task_type)

        if model_name not in models:
            raise ValueError(f"{model_name} not registered")

        return models[model_name]

    def list_models(self, task_type):
        return list(self.get_models(task_type).keys())


class Parameters:
    def __init__(self):
        self.regression = {
            "LR": lambda trial: {
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
                "positive": trial.suggest_categorical("positive", [True, False]),
            },
            "L1": lambda trial: {
                "alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True),
                "max_iter": trial.suggest_int("max_iter", 1000, 10000),
                "selection": trial.suggest_categorical(
                    "selection", ["cyclic", "random"]
                ),
            },
            "L2": lambda trial: {
                "alpha": trial.suggest_float("alpha", 1e-6, 100.0, log=True),
                "solver": trial.suggest_categorical(
                    "solver", ["auto", "svd", "cholesky", "lsqr", "sag", "saga"]
                ),
            },
            "EN": lambda trial: {
                "alpha": trial.suggest_float("alpha", 1e-6, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": trial.suggest_int("max_iter", 1000, 10000),
                "selection": trial.suggest_categorical(
                    "selection", ["cyclic", "random"]
                ),
            },
            "RF": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            },
            "BR": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 300),
                "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            },
            "GB": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            },
            "KNR": lambda trial: {
                "n_neighbors": trial.suggest_int("n_neighbors", 2, 50),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
                ),
                "p": trial.suggest_int("p", 1, 2),
            },
            "SVR": lambda trial: {
                "C": trial.suggest_float("C", 1e-3, 100, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-4, 1.0, log=True),
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf", "sigmoid"]
                ),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "degree": trial.suggest_int("degree", 2, 5),
            },
            "DT": lambda trial: {
                "max_depth": trial.suggest_int("max_depth", 2, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "criterion": trial.suggest_categorical(
                    "criterion", ["squared_error", "friedman_mse", "absolute_error"]
                ),
            },
        }
        self.classification = {
            "LR": lambda trial: {
                "C": trial.suggest_float("C", 1e-4, 100, log=True),
                "solver": trial.suggest_categorical(
                    "solver", ["lbfgs", "liblinear", "saga"]
                ),
                "max_iter": trial.suggest_int("max_iter", 100, 2000),
            },
            "RF": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            },
            "BC": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 10, 300),
                "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
                "max_features": trial.suggest_float("max_features", 0.5, 1.0),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            },
            "GB": lambda trial: {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.001, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            },
            "KNC": lambda trial: {
                "n_neighbors": trial.suggest_int("n_neighbors", 2, 50),
                "weights": trial.suggest_categorical(
                    "weights", ["uniform", "distance"]
                ),
                "algorithm": trial.suggest_categorical(
                    "algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
                ),
                "p": trial.suggest_int("p", 1, 2),
            },
            "SVC": lambda trial: {
                "C": trial.suggest_float("C", 1e-4, 100, log=True),
                "kernel": trial.suggest_categorical(
                    "kernel", ["linear", "poly", "rbf", "sigmoid"]
                ),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "degree": trial.suggest_int("degree", 2, 5),
            },
            "DT": lambda trial: {
                "max_depth": trial.suggest_int("max_depth", 2, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "criterion": trial.suggest_categorical(
                    "criterion", ["gini", "entropy", "log_loss"]
                ),
            },
        }

    def get_search_space(self, task_type, model_name):
        if task_type == "Regression":
            return self.regression[model_name]
        elif task_type == "Classification":
            return self.classification[model_name]
        else:
            return "Invalid Model Name"
