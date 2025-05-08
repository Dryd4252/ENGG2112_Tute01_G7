import pandas as pd
import numpy as np

import ml_models.mlp as mlp
from ml_models.rfr import RfrModel as rfr

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
    size_df.to_csv("results/test.csv")
    print(size_df)
    print(models)
    models[f"model_{(128, 64, 32)}"].create_graph()



def main():
    seed = 69420
    data = pd.read_csv("train.csv")
    save_files = True

    mlp_model = mlp.MlpModel(data, 50, seed=seed)
    rfr_model = rfr(data, seed=seed, track_training_time=False)

    ml_models = [mlp_model, rfr_model]
    
    for model in ml_models:
        model.process_data()
        model.train_model()
        model.make_prediction()
        model.classify_model_performance()
        print(model.get_statistics())
        model.create_graph(save_fig=save_files) # Saves graph if save_fig is True

        if save_files: # Savees stats if save_files is True
            model.save_statistics(model.name) 
            
    net_sizes = [(10,),(64,64),(128,64,32),(256,128,64)]
  
    # optimise_mlp(data,net_sizes,seed)

if __name__ == "__main__":
    main()
