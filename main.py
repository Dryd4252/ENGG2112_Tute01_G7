import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import randint, uniform

from ml_models.mlp import MlpModel as mlp
from ml_models.rfr import RfrModel as rfr
from ml_models.xgb import XgbModel as xgb

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

def rfr_test_optomiser(property_data: pd.DataFrame, seed: int):
    param_grid = {
        'n_estimators': randint(100, 300),
        'max_depth': [None] +  list(range(10, 51, 10)),
        'min_samples_split': randint(2, 5),
        'min_samples_leaf': randint(1, 2),
        'max_features': ['sqrt', 'log2']
    }

    rfr_model_t = rfr(property_data, seed=seed)
    rfr_model_t.process_data(["critical_temp"])
    rfr_model_t.optomise_model(param_grid)
    rfr_model_t.test_prediction()
    rfr_model_t.classify_model_performance()

    print(rfr_model_t.get_statistics())
    print(rfr_model_t.get_params())

def xgb_optimiser(property_data: pd.DataFrame, seed: int):
    param_grid = {
        'n_estimators': randint(100, 500),     # 100 - 1000
        'learning_rate': uniform(0.01, 0.07),    #0.01 - 0.1
        'max_depth': randint(4, 7),           #3 - 10
        'subsample': uniform(0.5, 0.5),          #0.5 - 1
        'colsample_bytree': uniform(0.5, 0.5),   #0.5 - 1
        'lambda': uniform(1, 2),               #0 - > 1
        'alpha': uniform(1, 2),                #0 - > 1
        'booster': ['gbtree']
    }

    xgb_model_t = xgb(property_data, seed=seed)
    xgb_model_t.process_data(["critical_temp"])
    xgb_model_t.optimise_model(param_grid)
    xgb_model_t.test_prediction()
    xgb_model_t.classify_model_performance()

    results = xgb_model_t.get_cv_results()

    # Train performance
    mse_train, rmse_train, mae_train, r2_train = xgb_model_t.evaluate_train_performance()

    # Extract cross-validation scores
    mean_val_scores = np.mean(results['mean_test_score'])
    mean_train_scores = results.get('mean_train_score')
    if mean_train_scores is not None:
        mean_train_scores = np.mean(mean_train_scores)

    # Test performance
    xgb_model_t.classify_model_performance()
    mse_test, rmse_test, mae_test, r2_test = xgb_model_t.get_statistics()

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
    print(xgb_model_t.get_params())
    return xgb_model_t

def main(save_files):
    seed = None
    property_data = pd.read_csv("train.csv")
    
    xgb_model_t = xgb_optimiser(property_data, seed)
    save_files = True

    if save_files: # Savees stats if save_files is True
        xgb_model_t.save_statistics(xgb_model_t.name) 

    return

    # net_sizes = [(10,),(64,64),(128,64,32),(256,128,64)]
    # optimise_mlp(property_data,net_sizes,seed)
    
    mlp_model_1 = mlp(property_data, 50, seed=seed)
    rfr_model_1 = rfr(property_data, seed=seed)
    xgb_model_1 = xgb(property_data, 100, 0.1, 5, seed=seed)

    ml_models_1 = [mlp_model_1, rfr_model_1, xgb_model_1]
    
    for model in ml_models_1:
        model.process_data(["critical_temp"])
        model.train_model()
        model.test_prediction()
        model.classify_model_performance()
        print(model.get_statistics())

        model.create_graph(save_fig=save_files) # Saves graph if save_fig is True
        if save_files: # Savees stats if save_files is True
            model.save_statistics(model.name) 


    symbol_data = pd.read_csv("unique_m.csv")

    symbol_drop_columns = ["critical_temp", "material"]
    symbol_data = symbol_data.drop(columns=symbol_drop_columns)

    property_symbol_data = pd.concat([property_data, symbol_data], axis=1)



if __name__ == "__main__":
    save_files = len(sys.argv) > 1 and sys.argv[1] is True
    main(save_files)

    plt.show()
