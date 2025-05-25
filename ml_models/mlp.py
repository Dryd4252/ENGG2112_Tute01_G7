import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import abstract_ml_model

class MlpModel(abstract_ml_model.AbstractMlModel):
    # im changing hl_sizes to a tuple to more easily change the variation of hidden layers, inputs will be lik (100,) or (128,64,32)
    def __init__(self, 
                 data: pd.DataFrame, 
                 hl_sizes = (10,), 
                 activation = "relu",
                 solver='adam', 
                 alpha=0.0001,
                 batch_size="auto",
                 learning_rate = "constant",
                 learning_rate_init=0.001,
                 max_iterations=10000, 
                 tol = 1e-4,
                 momentum = 0.9, 
                 nesterovs_momentum = True,
                 early_stopping=True, 
                 validation_fraction = 0.1,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 epsilon = 1e-8,
                 n_iter_no_change = 10,
                 seed=None):
        super().__init__(data, self.__class__.__name__, seed=seed)
        self.hl_sizes: tuple = hl_sizes 
        self.activation = activation
        self.solver = solver
        self.alpha = alpha 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iterations: int =  max_iterations
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping

    def process_data(self, target_label: list[str], split_size: float = 0.2,  exclude_features: list[str] = []) -> None:
        X = self.data.drop(columns=[*target_label, *exclude_features])
        y = self.data[target_label]

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
                batch_size = self.batch_size,
                learning_rate = self.learning_rate, 
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iterations, 
                tol = self.tol,
                momentum = self.momentum,
                nesterovs_momentum = self.nesterovs_momentum,
                validation_fraction = self.validation_fraction,
                beta_1 = self.beta_1,
                beta_2 = self.beta_2,
                epsilon = self.epsilon,
                n_iter_no_change = self.n_iter_no_change,
                random_state=self.seed,
                early_stopping=self.early_stopping,
                verbose=2
            ))
        ])
        
        if len(self.y_train.columns) == 1:
            self.model = mlp_model.fit(self.X_train, self.y_train.values.ravel())
        else:
            self.model = mlp_model.fit(self.X_train, self.y_train)

    def initalise_model(self) -> BaseEstimator:
        return MLPRegressor(random_state=self.seed)
    
    @abstract_ml_model.track_time
    def run_optimisation(self, param_grid, n_iter=20, target_label=["critical_temp"]):
        self.process_data(target_label)
        self.optimise_model(param_grid, n_iter=n_iter, verbose=1)
        self.test_prediction()
        self.classify_model_performance()

        results = self.get_cv_results()

        # Train performance
        mse_train, rmse_train, mae_train, r2_train = self.evaluate_train_performance()

        # Extract cross-validation scores
        mean_val_scores = np.mean(results['mean_test_score'])
        mean_train_scores = results.get('mean_train_score')
        if mean_train_scores is not None:
            mean_train_scores = np.mean(mean_train_scores)

        # Test performance
        mse_test, rmse_test, mae_test, r2_test = self.get_statistics()

        print("\n===== Cross-validation scores =====")
        print(f"Mean test scores: {mean_val_scores}")
        if mean_train_scores is not None:
            print(f"Mean train scores: {mean_train_scores}")

        print("\n===== Final Train Set Statistics =====")
        print(f"MSE:  {mse_train:.4f}")
        print(f"RMSE: {rmse_train:.4f}")
        print(f"MAE:  {mae_train:.4f}")
        print(f"R²:   {r2_train:.4f}")

        print("\n===== Final Test Set Statistics =====")
        print(f"MSE:  {mse_test:.4f}")
        print(f"RMSE: {rmse_test:.4f}")
        print(f"MAE:  {mae_test:.4f}")
        print(f"R²:   {r2_test:.4f}")

        print("\n===== Best Parameters =====")
        print(self.get_params())
    @abstract_ml_model.track_time
    def sweep_single_parameter(self, 
                           param_name: str, 
                           param_values: list, 
                           fixed_params: dict, 
                           target_label: list[str] = ["critical_temp"]) -> pd.DataFrame:
        """
        Sweep over a single hyperparameter while keeping others fixed.

        Parameters:
        - param_name: str, the name of the parameter to vary
        - param_values: list, values for the parameter
        - fixed_params: dict, base/fixed hyperparameters
        - target_label: list of target labels to predict

        Returns:
        - pd.DataFrame with metrics for each value of the varied parameter
        """
        self.process_data(target_label)
        results = []

        for i, value in enumerate(param_values, start=1):
            print(f"Iteration {i}/{len(param_values)} — testing {param_name} = {value}")
            # Set parameters
            param_set = fixed_params.copy()
            param_set[param_name] = value

            # Update self's hyperparameters dynamically
            for k, v in param_set.items():
                setattr(self, k, v)

            # Train the model
            self.train_model()
            self.test_prediction()
            self.classify_model_performance()

            # Test performance
            mse, rmse, mae, r2 = self.get_statistics()

            results.append({
                param_name: value,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            })
            # property_data = pd.read_csv("../train.csv")
            # for i in range(1, 10):
            #     print(f"starting {i}th iteration")
            #     mlp_model = MlpModel(property_data, seed=None)
            #     mlp_model.run_optimisation(param_grid=param_grid_adam, n_iter=5)
            #     mlp_model.save_statistics("mlp_test_4")
            #     print(f"ending {i}th iteration")

        return pd.DataFrame(results).set_index(param_name)    


def main():
    # net_sizes = [(10,),(64,64),(128,64,32),(256,128,64)]
    # optimise_mlp(property_data,net_sizes,seed)
    hidden_layer_sizes = [(10,),(64,64),(128,64,32),(256,128,64)]
    solvers = ['relu', 'tanh', 'logistic', 'identity']
    alpha = np.logspace(-6, 0, num=7)
    batch_size = (32, 64, 128, 256)
    learning_rate = ['constant', 'invscaling', 'adaptive']
    learning_rate_init = np.logspace(-4,-2,num = 4)
    beta_1 = np.arange(0.5,1.0,0.1)
    beta_2 = np.linspace(0.9,0.99999,5)
    n_iter_no_change = np.arange(5,20,1)
    tol = np.logspace(-4,-2,num = 4)
    max_iterations = [10000]
    param_grid_adam = {
        'hidden_layer_sizes': hidden_layer_sizes,
        'activation': solvers,
        'alpha': alpha,
        'batch_size': batch_size,
        'beta_1': beta_1,
        'beta_2':beta_2,
        'n_iter_no_change': n_iter_no_change,
        'tol':tol,
        'max_iter': max_iterations,
        'verbose': [2] 
    }
    param_grid_SGD = {
        'activation': solvers,
        'alpha': alpha,
        'learning_rate': learning_rate,
        'learning_rate_init': learning_rate_init,

    }
    property_data = pd.read_csv("../train.csv")

    # model = MlpModel(property_data,seed=None)
    # model.process_data(target_label=["critical_temp"])
    # model.train_model()
    # model.test_prediction()
    # model.classify_model_performance()
    # print(model.get_params())
   

if __name__ == "__main__":
    main()

