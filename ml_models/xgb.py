import pandas as pd
import numpy as np

from xgboost import XGBRegressor

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score

from . import abstract_ml_model

class XgbModel(abstract_ml_model.AbstractMlModel):
    def __init__(
            self, 
            data: pd.DataFrame,
            n_estimators: int, 
            learning_rate: float,
            max_depth: int,
            seed=None
        ):
        super().__init__(data, self.__class__.__name__, seed=seed)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def process_data(self,target_label, test_size=0.2, exclude_features: list[str] = []):
        X = self.data.drop(columns=[*target_label, *exclude_features])
        y = self.data[target_label]

        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(X, y, test_size=test_size, random_state=self.seed)

    def train_model(self) -> None:

        # configure parameters and hyperparameters
        xgb_model = XGBRegressor(
            n_estimators=self.n_estimators,     # number of boosting rounds
            learning_rate=self.learning_rate,    # step size shrinkage
            max_depth=self.max_depth,          # depth of each tree
            random_state=self.seed       # reproducibility
        )

        # making pipeline
        steps = [
            ('scaler', StandardScaler()),
            ('model', xgb_model)
        ]

        # train and test the model
        if len(self.y_train.columns) == 1:
            self.model = Pipeline(steps).fit(self.X_train, self.y_train.values.ravel())
        else:
            self.model = Pipeline(steps).fit(self.X_train, self.y_train)

    def initalise_model(self) -> BaseEstimator:
        return XGBRegressor(random_state=self.seed)
