import pandas as pd
import ml_models
import numpy as np
import matplotlib.pyplot as plt

import ml_models.xgb as xgb
import ml_models.mlp as mlp

## Function to optimise mlp based on certain parameteres
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

def optimise_mlp(data,net_sizes,seed):
    size_scores = []
    i = 0
    alpha = 0.0001
    models = {}
    ## Run Size Study, vary between a few 
    for net_size in net_sizes:
        i += 1
        print(f"running test #{i} for size parameter tuple {net_size}")
        mlp_model = mlp.MlpModel(data, net_size,alpha = alpha, seed=seed)
        mlp_model.process_data()
        mlp_model.train_model()
        mlp_model.make_prediction()
        mlp_model.classify_model_performance()
        size_scores.append(mlp_model.get_statistics())
        models[f"model_{net_size}"] = mlp_model
        print("done")
    
    size_df = pd.DataFrame(size_scores, columns= ["mse", "rmse", "mae","r2"], index = net_sizes)
    size_df["weighted_score"] = size_df.apply(weighted_score, axis=1,args=(size_df,))
    print(size_df)
    print(models)
    print(models[f"model_{(128, 64, 32)}"].create_graph())

def main():
    seed = 6969
    property_data = pd.read_csv("train.csv")

    net_sizes = [(10,),(64,64),(128,64,32),(256,128,64)]
    # optimise_mlp(property_data,net_sizes,seed)

#    mlp_model = mlp.MlpModel(property_data, (10,), alpha = 0.0001, seed=seed)
#    mlp_model.process_data()
#    mlp_model.train_model()
#    mlp_model.make_prediction()
#    mlp_model.classify_model_performance()
#    print(mlp_model.get_statistics())
#    mlp_model.create_graph()

    xgb_model = xgb.XgbModel(property_data, 100, 0.1, 5, seed=seed)
    xgb_model.process_data(["critical_temp"])
    xgb_model.train_model()
    xgb_model.make_prediction()
    xgb_model.classify_model_performance()
    print(xgb_model.get_statistics())
    xgb_model.create_graph()

    symbol_data = pd.read_csv("unique_m.csv")

    symbol_drop_columns = ["critical_temp", "material"]
    symbol_data = symbol_data.drop(columns=symbol_drop_columns)

    property_symbol_data = pd.concat([property_data, symbol_data], axis=1)

    xgb_model = xgb.XgbModel(property_symbol_data, 100, 0.1, 5, seed=seed)
    xgb_model.process_data(["critical_temp"])
    xgb_model.train_model()
    xgb_model.make_prediction()
    xgb_model.classify_model_performance()
    print(xgb_model.get_statistics())
    xgb_model.create_graph()


if __name__ == "__main__":
    main()
    plt.show()
