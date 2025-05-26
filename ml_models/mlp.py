import pandas as pd
import numpy as np
import os

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import abstract_ml_model

class MlpModel(abstract_ml_model.AbstractMlModel):
    # im changing hl_sizes to a tuple to more easily change the variation of hidden layers, inputs will be lik (100,) or (128,64,32)
    def __init__(self, 
                 data: pd.DataFrame, 
                 hidden_layer_sizes = (10,), 
                 activation = "relu",
                 solver='adam', 
                 alpha=0.0001,
                 batch_size="auto",
                 learning_rate = "constant",
                 learning_rate_init=0.001,
                 max_iter=10000, 
                 tol = 1e-4,
                 momentum = 0.9, 
                 nesterovs_momentum = True,
                 early_stopping=True, 
                 validation_fraction = 0.1,
                 beta_1 = 0.9,
                 beta_2 = 0.999,
                 epsilon = 1e-8,
                 n_iter_no_change = 10,
                 seed=None,
                 power_t = 0.5,
                 verbose = 1
                 ):
        super().__init__(data, self.__class__.__name__, seed=seed)
        self.hidden_layer_sizes: tuple = hidden_layer_sizes 
        self.activation = activation
        self.solver = solver
        self.alpha = alpha 
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter: int =  max_iter
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.power_t = power_t
    def process_data(self, target_label: list[str], split_size: float = 0.2,  exclude_features: list[str] = []) -> None:
        X = self.data.drop(columns=[*target_label, *exclude_features])
        y = self.data[target_label]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
    
    @abstract_ml_model.track_time
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
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver=self.solver,
                alpha=self.alpha,
                batch_size = self.batch_size,
                learning_rate = self.learning_rate, 
                learning_rate_init=self.learning_rate_init,
                max_iter=self.max_iter, 
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
                power_t = self.power_t,
                verbose= self.verbose
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
    def plot_parameter_sweep(self,results_df, param_name='hidden_layer_sizes',file_name = "test"):
        """
        Plot MSE, RMSE, MAE, and R² against different hidden layer configurations.

        Parameters:
        - results_df: pd.DataFrame from sweep_single_parameter
        - param_name: str, name of the varied parameter (should be index of DataFrame)
        """
        if not isinstance(results_df.index[0], str):
            results_df.index = results_df.index.map(str)  # Convert tuples to strings

        metrics = ['mse', 'rmse', 'mae', 'r2']
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        x_labels = results_df[param_name].tolist()  # Actual param values like tuples
        print(x_labels)
        for i, metric in enumerate(metrics):
            ax = axes[i]
            results_df[metric].plot(kind='bar', ax=ax)
            ax.set_title(metric.upper())
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(results_df.index)))
            ax.set_xticklabels([str(label) for label in x_labels], rotation=45, ha='right')
            y_min = results_df[metric].min()
            y_max = results_df[metric].max()
            margin = 0.05 * (y_max - y_min)  # 5% margin

            ax.set_ylim(y_min - margin, y_max + margin)

        plt.tight_layout()
        plt.suptitle(f"Performance Metrics vs. {param_name}", fontsize=16, y=1.03)
        plt.savefig(f"results/graph/{file_name}_metrics.png")
        plt.show()


def main():
    # net_sizes = [(10,),(64,64),(128,64,32),(256,128,64)]
    # optimise_mlp(property_data,net_sizes,seed)
    hidden_layer_sizes = [(10,),(64,64),(128,64,32),(256,128,64)]
    hidden_layer_sizes_2 = [(256,256,128),(256,256,256),(512,256,128),(512,256,256)]
    hidden_layer_sizes_3 = [(1024,256,128),(1024,256,256),(1024,512,256),(1024,512,512)]
    hidden_layer_sizes_4 = [(256,128,64,32),(256,256,128,64),(512,256,256,128),(1024,512,256,128)]

    hidden_layer_sizes_5 = [(128,64,32,16,8),(256,128,64,32,16),(256,128,64,64,32),(256,128,128,64,64)]
    hidden_layer_sizes_6 = [(512,256,128,64,64)]
    hidden_layer_sizes_7 = [(1024,512,265,128,64)]
    hidden_layer_sizes_8 = [(1024,512,512,256,128)]


    solvers = ['relu', 'tanh']
    alpha = np.logspace(-6, -1, num=6)
    batch_size = [32, 64, 128, 256, 512, 1024]
    learning_rate = ['constant', 'invscaling', 'adaptive']
    learning_rate_init = np.logspace(-4,-1.5,num = 10)
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
        'n_iter_no_change': [10],
        'tol':[1e-4],
        'max_iter': [10000],
        'verbose': [False] 
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
   
    adam_fixed_params = {
        'hidden_layer_sizes': (1024, 512, 256, 128),#(512,256,256),
        'activation': 'relu',
        'learning_rate_init': 0.002,
        'alpha': 0.01,
        'batch_size': 'auto',
        'beta_1': 0.9,
        'beta_2':0.999,
        'n_iter_no_change': 15,
        'tol': 1e-5,
        'max_iter': 10000,
        'verbose': 0
    }

    
    def test_params(param_name,test_vals,file_name):
        mlp_model = MlpModel(property_data, seed=42)
        # test_vals = hidden_layer_sizes_8
        results_df = mlp_model.sweep_single_parameter(
            param_name,
            param_values=test_vals,
            fixed_params=adam_fixed_params
        )
        print(results_df)
        name = file_name
        results_df.to_csv(f"results/txt/{name}.csv", mode = 'a', header=not os.path.exists(f"results/txt/{name}.csv"), index=True)


        df = pd.read_csv(f"results/txt/{name}.csv")

        mlp_model.plot_parameter_sweep(df, param_name,file_name)
    param_name="batch_size"
    test_vals = batch_size
    file_name = 'sequential_batch_size_(1024, 512, 256, 128)'
    test_params(param_name,test_vals,file_name)
    # name = file_name
    # df = pd.read_csv(f"results/txt/{name}.csv")
    # mlp_model = MlpModel(property_data, seed=42)

    # mlp_model.plot_parameter_sweep(df, param_name,file_name)
   
    
if __name__ == "__main__":
    main()

 # mlp_model = MlpModel(property_data,**adam_fixed_params)
    # mlp_model.process_data(target_label = ["critical_temp"])
    # mlp_model.train_model()
    # mlp_model.test_prediction()
    # mlp_model.classify_model_performance()

    # mse_train, rmse_train, mae_train, r2_train = mlp_model.evaluate_train_performance()
    # mse_test, rmse_test, mae_test, r2_test = mlp_model.get_statistics()

    # print("train")
    # print(mse_train, rmse_train, mae_train, r2_train)
    # print("test")
    # print(mse_test, rmse_test, mae_test, r2_test)