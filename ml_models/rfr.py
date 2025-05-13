# @file: ml_models/rfr.py
# @description: This file contains the Random Forest Regressor model class. 
# @author: Oliver Cook

# Imports
# =============================================================================
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

from .abstract_ml_model import AbstractMlModel, track_time

# Class Definition
# =============================================================================
class RfrModel(AbstractMlModel):

    # Constructor
    def __init__(self, data: pd.DataFrame, n_estimators=141, max_depth=30, min_samples_split=3, min_samples_leaf=1, max_features='sqrt', n_jobs=-1, seed=None) -> None:
        super().__init__(data, self.__class__.__name__, seed=seed)
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.min_samples_split=min_samples_split
        self.min_samples_leaf=min_samples_leaf
        self.max_features=max_features
        self.n_jobs=n_jobs

    # Process data definition
    def process_data(self, target_label: list[str], split_size: float = 0.2, exclude_features: list[str] = []) -> None:
        X = self.data.drop(columns=[*target_label, *exclude_features])
        y = self.data[target_label]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=split_size, random_state=self.seed)

    # tain model definition
    @track_time
    def train_model(self) -> None:
        numerical_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ("scaler", MinMaxScaler(), numerical_features)
            ]
        )

        # Random Forest Regressor pipeline
        rfr_model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            (
                'rfr', RandomForestRegressor
                (
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=self.max_features,
                    random_state=self.seed,
                    n_jobs=self.n_jobs
                )
            )
        ])

        self.model = rfr_model.fit(self.X_train, self.y_train.values.ravel())
