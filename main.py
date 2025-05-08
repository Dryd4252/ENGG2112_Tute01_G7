import pandas as pd
import ml_models

import ml_models.mlp as mlp
from ml_models.rfr import RfrModel as rfr

def main():
    seed = 69420
    data = pd.read_csv("train.csv")
    save_files = True

    mlp_model = mlp.MlpModel(data, 50, seed=seed)
    rfr_model = rfr(data, seed=seed)

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

if __name__ == "__main__":
    main()
