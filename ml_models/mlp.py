import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from . import abstract_ml_model

class MlpModel(abstract_ml_model.AbstractMlModel):
    # im changing hl_sizes to a tuple to more easily change the variation of hidden layers, inputs will be lik (100,) or (128,64,32)
    def __init__(self, data: pd.DataFrame, hl_sizes: tuple, activation = "relu",solver='adam', alpha=0.0001,learning_rate_init=0.001,early_stopping=False, max_iterations=10000, seed=None):
        super().__init__(data, self.__class__.__name__, seed=seed)
        self.max_iterations: int =  max_iterations
        self.hl_sizes: tuple = hl_sizes 
        self.activation = activation
        self.solver = solver
        self.alpha = alpha 
        self.learning_rate_init = learning_rate_init
        self.early_stopping = early_stopping

    def process_data(self, exclude_features: list[str] = []) -> None:
        label = "critical_temp"
        X = self.data.drop(columns=[label, *exclude_features])
        y = self.data[label]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)

    def train_model(self) -> None:
        numerical_features = self.X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = self.X_train.select_dtypes(include=['object']).columns.tolist() 

        mlp_preprocessor = ColumnTransformer(
            transformers=[
                ("categorical",OneHotEncoder(handle_unknown='ignore'),categorical_features),
                ("scaler",StandardScaler(), numerical_features)
            ]
        )

        mlp_model = Pipeline(steps=[
            ('preprocessor',mlp_preprocessor),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=self.hl_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iterations, 
                random_state=self.seed,
                early_stopping=self.early_stopping,
                verbose=False
            ))
        ])

        self.model = mlp_model.fit(self.X_train, self.y_train.values.ravel())

