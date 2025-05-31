import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score

from . import abstract_ml_model
from .abstract_ml_model import require_state, ModelState

def plot_property_importance(model, property_data):
    feature_importance = model.get_feature_importances()
    property_cols = property_data.drop(columns=["critical_temp"]).columns
    importance_series = pd.Series(feature_importance[:len(property_cols)], index=property_cols)
    sorted_importance = importance_series.sort_values(ascending=False)[:10]

    plt.figure(figsize=(10, 6))
    sorted_importance[::-1].plot(kind="barh", color="skyblue")
    plt.title("Top 10 Most Important Properties in Predicting Tc")
    plt.xlabel("XGBoost Gain Importance")
    plt.tight_layout()

def plot_element_importance(model, symbol_data):
    feature_importance = model.get_feature_importances()
    element_cols = symbol_data.columns
    offset = -len(element_cols)  # features are appended to property_data

    importance_series = pd.Series(feature_importance[offset:], index=element_cols)
    sorted_importance = importance_series.sort_values(ascending=False)[:10]

    plt.figure(figsize=(10, 6))
    sorted_importance[::-1].plot(kind="barh", color="lightcoral")
    plt.title("Top 10 Most Important Elements in Predicting Tc")
    plt.xlabel("XGBoost Gain Importance")
    plt.tight_layout()

def get_top_important_features(model, top_n=10):
    importance = model.model.feature_importances_
    feature_names = model.feature_names
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=False)
    return importance_df.head(top_n)['feature'].tolist()

class XgbModel(abstract_ml_model.AbstractMlModel):
    def __init__(
            self, 
            data: pd.DataFrame,
            n_estimators: int = None, 
            learning_rate: float = None,
            max_depth: int = None,
            subsample = None,
            colsample_bytree = None,
            lambd = None,
            alpha = None,
            booster = None,
            seed=None):
        super().__init__(data, self.__class__.__name__, seed=seed)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.lambd = lambd
        self.alpha = alpha
        self.booster = booster

    def process_data(self, target_label, test_size=0.2, exclude_features: list[str] = []):
        X = self.data.drop(columns=[*target_label, *exclude_features])
        y = self.data[target_label]

        (self.X_train, self.X_test, self.y_train, self.y_test) = train_test_split(X, y, test_size=test_size, random_state=self.seed)

    def train_model(self) -> None:

        # configure parameters and hyperparameters
        xgb_model = XGBRegressor(
            n_estimators=self.n_estimators,     # number of boosting rounds
            learning_rate=self.learning_rate,    # step size shrinkage
            max_depth=self.max_depth,          # depth of each tree
            subsample = self.subsample ,
            colsample_bytree = self.colsample_bytree,
            reg_lambda = self.lambd,
            alpha = self.alpha,
            booster = self.booster,
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
    
    @require_state(ModelState.MODEL_TRAINED)
    def get_feature_importances(self) -> np.ndarray:
        if hasattr(self.model, "named_steps"):
            xgb_model_only = self.model.named_steps["model"]
        else:
            xgb_model_only = self.model

        return xgb_model_only.feature_importances_

