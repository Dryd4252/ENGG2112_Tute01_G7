import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score

from abc import ABC, ABCMeta, abstractmethod
from enum import Enum

class ModelState(Enum):
    INIT = 1
    DATA_PROCESSED = 2
    MODEL_TRAINED = 3
    PREDICTION_MADE = 4
    CLASSIFIED_PERFORMANCE = 5

def require_state(required_state: ModelState):
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            if self._state.value < required_state.value:
                raise RuntimeError(
                    f"{method.__name__} can only be called in state {required_state.name}. "
                    f"The current state is {self._state.name}"
                )
            return method(self, *args, **kwargs)
        return wrapper
    return decorator

def transition_state(next_state: ModelState):
    def decorator(method):
        def wrapper(self, *args, **kwargs):
            self._state = next_state
            return method(self, *args, **kwargs)
        return wrapper
    return decorator

def track_time(method):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)
        end = time.time()
        print(f"{method.__qualname__} execute time: {(end - start):.5f}s")
        return result
    return wrapper

class AbstractMlModelMeta(ABCMeta):

    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # key: Method name, value: (Required state, Next State)
        method_states = {
            "process_data": (ModelState.INIT, ModelState.DATA_PROCESSED),
            "train_model": (ModelState.DATA_PROCESSED, ModelState.MODEL_TRAINED)
        }

        for method_name, (required_state, next_state) in method_states.items():
            method = namespace[method_name]
            decorated = require_state(required_state)(transition_state(next_state)(method))
            setattr(cls, method_name, decorated)

        return cls

class AbstractMlModel(ABC, metaclass=AbstractMlModelMeta):
    
    def __init__(self, data: pd.DataFrame, name: str, seed=None):
        self._state = ModelState.INIT
        self.seed = seed

        self.data: pd.DataFrame = data
        self.name = name

    @abstractmethod
    def process_data(self) -> None:
        pass

    @abstractmethod
    def train_model(self) -> None:
        pass

    @abstractmethod
    def initalise_model(self) -> BaseEstimator:
        pass

    @require_state(ModelState.DATA_PROCESSED)
    @transition_state(ModelState.MODEL_TRAINED)
    def optimise_model(self, param_grid: dict, n_iter=5, cv=5, scoring="r2", verbose=2) -> None:

        # Set up RandomSearchCV
        self.random_search = RandomizedSearchCV(
            estimator=self.initalise_model(),
            param_distributions=param_grid,
            n_iter=n_iter,  
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.seed,
            verbose=verbose,
            return_train_score=True
        )

        # Run ranndom search
        if len(self.y_train.columns) == 1:
            self.random_search.fit(self.X_train, self.y_train.values.ravel())
        else:
            self.random_search.fit(self.X_train, self.y_train)

        # Get best model
        self.model = self.random_search.best_estimator_

    @require_state(ModelState.MODEL_TRAINED)
    @transition_state(ModelState.PREDICTION_MADE)
    def test_prediction(self) -> None:
        self.y_pred = self.model.predict(self.X_test)

    @require_state(ModelState.MODEL_TRAINED)
    def make_prediction(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(features)

    @require_state(ModelState.MODEL_TRAINED)
    @transition_state(ModelState.CLASSIFIED_PERFORMANCE)
    def classify_model_performance(self) -> None:
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = np.sqrt(self.mse)
        self.mae = mean_absolute_error(self.y_test, self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred)

        self.mse_train, self.rmse_train, self.mae_train, self.r2_train = self.evaluate_train_performance()
    
    @require_state(ModelState.MODEL_TRAINED)
    def evaluate_train_performance(self):
        y_train_pred = self.model.predict(self.X_train)
        mse_train = mean_squared_error(self.y_train, y_train_pred)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(self.y_train, y_train_pred)
        r2_train = r2_score(self.y_train, y_train_pred)
        return mse_train, rmse_train, mae_train, r2_train

    @require_state(ModelState.CLASSIFIED_PERFORMANCE)
    def get_statistics(self) -> None:
        return self.mse, self.rmse, self.mae, self.r2

    @require_state(ModelState.CLASSIFIED_PERFORMANCE)
    def save_statistics(self, name: str) -> None:
        os.makedirs("results/txt", exist_ok=True)

        results_df = pd.DataFrame({
            "mse": [self.mse],
            "rmse": [self.rmse],
            "mae": [self.mae],
            "r2": [self.r2],
            "train_mse": [self.mse_train],
            "train_rmse": [self.rmse_train],
            "train_mae": [self.mae_train],
            "train_r2": [self.r2_train],
        })

        results_df.to_csv(f"results/txt/{name}.csv", mode = 'a', header=not os.path.exists(f"results/txt/{name}.csv"), index=False)

        params = self.model.get_params()
        params_df = pd.DataFrame([params])
        params_df.to_csv(f"results/txt/{name}_params.csv", mode='a', header=not os.path.exists(f"results/txt/{name}_params.csv"), index=False)

    @require_state(ModelState.MODEL_TRAINED)
    def create_graph(self, save_fig=False) -> None:
        os.makedirs("results/graphs", exist_ok=True)
        # Scatter plot of actual vs predicted
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.y_pred, alpha=0.5, color='teal')
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')  # Perfect prediction line
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{self.name}: Actual vs Predicted')
        plt.grid(True)
        plt.tight_layout()

        if save_fig:
            plt.savefig(f"results/graphs/{self.name} model.png")

    @require_state(ModelState.MODEL_TRAINED)
    def get_params(self):
        return self.model.get_params()
    
    @require_state(ModelState.MODEL_TRAINED)
    def get_cv_results(self):
        return self.random_search.cv_results_


def plot_top_properties_boxplots(model, data, target_col="critical_temp", top_n=50, top_features=10):
    # use model to find predictions and add to dataframe
    X_test = model.X_test.copy()
    preds = model.make_prediction(X_test)
    X_test["predicted_Tc"] = preds
    
    # get the top n predicted temperatures
    top_data = X_test.nlargest(top_n, "predicted_Tc")
    
    # get all features
    features = [col for col in data.columns if col != target_col]
    
    # rank features using the mean values in top_data (dataframe with the top predicted Tc)
    top_features_list = top_data[features].mean().sort_values(ascending=False).head(top_features).index.tolist()
    
    # plot
    plt.figure(figsize=(15, 8))
    top_data.boxplot(column=top_features_list)
    plt.title(f"Distribution of Top {top_features} Properties in Top {top_n} Predicted Tc Superconductors")
    plt.xticks(rotation=45)
    medians = top_data[top_features_list].median()
    positions = range(1, len(top_features_list) + 1)
    for pos, feature in zip(positions, top_features_list):
        median_val = medians[feature]
        plt.text(pos, median_val, f'{median_val:.2f}', horizontalalignment='center',
                 verticalalignment='bottom', fontsize=9, color='blue')

    plt.tight_layout()
    plt.show()