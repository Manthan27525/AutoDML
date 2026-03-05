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

models = {
    "Regression": {
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
    },
    "Classification": {
        "LR": LogisticRegression,
        "RF": RandomForestClassifier,
        "BC": BaggingClassifier,
        "GB": GradientBoostingClassifier,
        "KNC": KNeighborsClassifier,
        "SVC": SVC,
        "DT": DecisionTreeClassifier,
    },
}


class Models:
    @staticmethod
    def get_models():
        return models
