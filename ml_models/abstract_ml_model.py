import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

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

        # self.X_train: pd.DataFrame = None
        # self.y_train: pd.DataFrame = None
        # self.X_test: pd.DataFrame = None
        # self.y_test: pd.DataFrame = None
        # self.y_pred: pd.DataFrame = None

        # self.model = None

        # self.mse: float = None
        # self.rmse: float = None
        # self.mae: float = None
        # self.r2: float = None

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
    def optomise_model(self, param_grid: dict, n_iter=5, cv=5, scoring="neg_mean_squared_error", verbose=0) -> None:

        # Set up GridSearchCV
        random_search = RandomizedSearchCV(
            estimator=self.initalise_model(),
            param_distributions=param_grid,
            n_iter=n_iter,  
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=self.seed,
            verbose=verbose
        )

        # Run grid search
        if len(self.y_train.columns) == 1:
            random_search.fit(self.X_train, self.y_train.values.ravel())
        else:
            random_search.fit(self.X_train, self.y_train)

        # Get best model
        self.model = random_search.best_estimator_

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
    
    @require_state(ModelState.CLASSIFIED_PERFORMANCE)
    def get_statistics(self) -> None:
        return self.mse, self.rmse, self.mae, self.r2

    @require_state(ModelState.CLASSIFIED_PERFORMANCE)
    def save_statistics(self, name: str) -> None:
        results_df = pd.DataFrame({
            "mse": [self.mse],
            "rmse": [self.rmse],
            "mae": [self.mae],
            "r2": [self.r2]
        })
        results_df.to_csv(f"results/txt/{name}.csv", index=False)

    @require_state(ModelState.MODEL_TRAINED)
    def create_graph(self, save_fig=False) -> None:
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
