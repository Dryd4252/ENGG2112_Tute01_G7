import pandas as pd
import ml_models

import ml_models.mlp as mlp
import ml_models.xgb as xgb

def main():
    seed = 6969
    data = pd.read_csv("train.csv")

    mlp_model = mlp.MlpModel(data, 50, seed=seed)
    mlp_model.process_data()
    mlp_model.train_model()
    mlp_model.make_prediction()
    mlp_model.classify_model_performance()
    print(mlp_model.get_statistics())
    mlp_model.create_graph()

    xgb_model = xgb.XgbModel(data, 100, 0.1, 5, seed=seed)
    xgb_model.process_data()
    xgb_model.train_model()
    xgb_model.make_prediction()
    xgb_model.classify_model_performance()
    print(xgb_model.get_statistics())
    xgb_model.create_graph()

if __name__ == "__main__":
    main()
