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
    def __init__(self, data: pd.DataFrame, seed=None) -> None:
        super().__init__(data, self.__class__.__name__, seed=seed)

    # Process data definition
    def process_data(self, split_size: float = 0.2, exclude_features: list[str] = []) -> None:
        target_label = "critical_temp"
        X = self.data.drop(columns=[target_label, *exclude_features])
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
