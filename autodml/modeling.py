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
from sklearn.metrics import r2_score, f1_score

from config.models import Models

from autodml.preprocessing import Preprocessor
import numpy as np
import pandas as pd

models = Models.get_models()


class ModelTrainer:
    def __init__(self, problem_type, x_train, x_test, y_train, y_test):
        self.models = models
        self.model_score = None
        self.best_model = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.problem = problem_type

    def train(self):
        print("Starting Training")
        model_score = {}

        if self.problem == "Regression":
            for name, model_class in self.models[self.problem].items():
                model = model_class()
                model.fit(self.x_train, self.y_train)
                pred = model.predict(self.x_test)
                model_score[name] = r2_score(self.y_test, pred)

        elif self.problem == "Classification":
            for name, model_class in self.models[self.problem].items():
                model = model_class()
                model.fit(self.x_train, self.y_train)
                pred = model.predict(self.x_test)
                model_score[name] = f1_score(self.y_test, pred, average="weighted")
        else:
            raise ValueError("Invalid Problem Type")

        self.model_score = model_score

        print("Training Completed")

    def best_model_selection(self):
        print("Selecting Best Model")

        best_score = -np.inf
        best_model = None

        for name, result in self.model_score.items():
            if result > best_score:
                best_score = result
                best_model = name

        self.best_model = best_model

        print("Best Model Selected")

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
