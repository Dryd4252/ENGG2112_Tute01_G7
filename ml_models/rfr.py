# @file: ml_models/rfr.py
# @description: This file contains the Random Forest Regressor model class. 
# @author: Oliver Cook

# Imports
# =============================================================================
import time
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler

from .abstract_ml_model import AbstractMlModel

# Class Definition
# =============================================================================
class RfrModel(AbstractMlModel):

    # Constructor
    def __init__(self, data: pd.DataFrame, seed=None, track_training_time: bool = True) -> None:
        super().__init__(data, self.__class__.__name__, seed=seed)
        self.track_training_time = track_training_time

    # Process data definition
    def process_data(self, split_size: float = 0.2, exclude_features: list[str] = []) -> None:
        target_label = "critical_temp"
        X = self.data.drop(columns=[target_label, *exclude_features])
        y = self.data[target_label]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=split_size, random_state=self.seed)

    # tain model definition
    def train_model(self) -> None:
        if self.track_training_time:
            time_start = time.time()
        
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
                    n_estimators=141,
                    max_depth=30,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    random_state=self.seed,
                    n_jobs=-1
                )
            )
        ])

        self.model = rfr_model.fit(self.X_train, self.y_train.values.ravel())
        if self.track_training_time:
            self.training_time = time.time() - time_start
            print(f"RFR training time: {self.training_time:.2f} seconds")
