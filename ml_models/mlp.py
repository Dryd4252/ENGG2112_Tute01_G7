import pandas as pd

from sklearn.base import BaseEstimator
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
    def __init__(self, 
                 data: pd.DataFrame, 
                 hl_sizes: tuple, 
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
                 early_stopping=False, 
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
                verbose=False
            ))
        ])
        
        if len(self.y_train.columns) == 1:
            self.model = mlp_model.fit(self.X_train, self.y_train.values.ravel())
        else:
            self.model = mlp_model.fit(self.X_train, self.y_train)

    def initalise_model(self) -> BaseEstimator:
        return MLPRegressor(random_state=self.seed)


def normalise(value, mean, std):
    z = (value - mean) / std
    # Rescale z-scores to 0–1 using sigmoid-like transformation
    return 1 / (1 + np.exp(-z))  # squashes into (0, 1) smoothly

def weighted_score(row, df):
    scaled_mse  = 1 - normalise(row["mse"],  df["mse"].mean(),  df["mse"].std())
    scaled_rmse = 1 - normalise(row["rmse"], df["rmse"].mean(), df["rmse"].std())
    scaled_mae  = 1 - normalise(row["mae"],  df["mae"].mean(),  df["mae"].std())
    scaled_r2   = normalise(row["r2"],       df["r2"].mean(),   df["r2"].std())

    return 0.25 * scaled_mse + 0.25 * scaled_rmse + 0.25 * scaled_mae + 0.25 * scaled_r2

## Function to optimise mlp based on certain parameteres
def optimise_mlp(data,net_sizes,seed):
    size_scores = []
    i = 0
    alpha = 0.0001
    models = {}
    ## Run Size Study, vary between a few 
    for net_size in net_sizes:
        i += 1
        print(f"running test #{i} for size parameter tuple {net_size}")
        mlp_model = mlp(data, net_size,alpha = alpha, seed=seed)
        mlp_model.process_data()
        mlp_model.train_model()
        mlp_model.test_prediction()
        mlp_model.classify_model_performance()
        size_scores.append(mlp_model.get_statistics())
        models[f"model_{net_size}"] = mlp_model
        print("done")
    
    size_df = pd.DataFrame(size_scores, columns= ["mse", "rmse", "mae","r2"], index = net_sizes)
    size_df["weighted_score"] = size_df.apply(weighted_score, axis=1,args=(size_df,))
    size_df.to_csv("results/test.csv")
    print(size_df)
    print(models)
    models[f"model_{(128, 64, 32)}"].create_graph()

def main():
    # net_sizes = [(10,),(64,64),(128,64,32),(256,128,64)]
    # optimise_mlp(property_data,net_sizes,seed)
    return

if __name__ == "__main__":
    main()

