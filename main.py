import pandas as pd
import ml_models

import ml_models.mlp as mlp

def main():
    seed = 6969
    data = pd.read_csv("train.csv")

    mlp_model = mlp.MlpModel(data, (128,64,32), seed=seed)
    mlp_model.process_data()
    mlp_model.train_model()
    mlp_model.make_prediction()
    mlp_model.classify_model_performance()
    print(mlp_model.get_statistics())
    mlp_model.create_graph()

    

if __name__ == "__main__":
    main()
