import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
from scipy.stats import randint, uniform
from sklearn.inspection import permutation_importance
from packaging.version import parse as parse_version

from ml_models.mlp import MlpModel as mlp
from ml_models.rfr import RfrModel as rfr
from ml_models.xgb import XgbModel as xgb
from ml_models.xgb import plot_property_importance, plot_element_importance
from ml_models.abstract_ml_model import plot_top_properties_boxplots


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
    rfr_model_t.optimise_model(param_grid)
    rfr_model_t.test_prediction()
    rfr_model_t.classify_model_performance()

    print(rfr_model_t.get_statistics())
    print(rfr_model_t.get_params())

def xgb_optimiser(property_data: pd.DataFrame, seed: int):
    param_grid_test = {
        'n_estimators': randint(100, 500),     # 100 - 1000
        'learning_rate': uniform(0.01, 0.07),    #0.01 - 0.1
        'max_depth': randint(4, 7),           #3 - 10
        'subsample': uniform(0.5, 0.5),          #0.5 - 1
        'colsample_bytree': uniform(0.5, 0.5),   #0.5 - 1
        'lambda': uniform(1, 2),               #0 - > 1
        'alpha': uniform(1, 2),                #0 - > 1
        'booster': ['gbtree']
    }

    param_grid = {
        'n_estimators': [323],     # 100 - 1000
        'learning_rate': [0.044994882],    #0.01 - 0.1
        'max_depth': [4],           #3 - 10
        'subsample': [0.500422911],          #0.5 - 1
        'colsample_bytree': [0.55965652],   #0.5 - 1
        'lambda': [2.333191548],               #0 - > 1
        'alpha': [3.922654256],                #0 - > 1
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

    ### Sub Problem A
    
    # Data processing
    property_data = pd.read_csv("train.csv")

    # Sub-Problem A

    print("Calling xgb_optimiser() for the first time to train model")
    xgb_model_t = xgb_optimiser(property_data, seed)
    # plot_property_importance(xgb_model_t, property_data
    plot_top_properties_boxplots(xgb_model_t, property_data)

    # for i in range(2):
    #     xgb_model_t = xgb_optimiser(property_data, seed)

    #     if save_files: # Savees stats if save_files is True
    #         xgb_model_t.save_statistics(xgb_model_t.name) 

    # rfr_test_optomiser(property_data, seed)

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

    ### Sub Problem B

    # Data processing
    symbol_data = pd.read_csv("unique_m.csv")
    symbol_drop_columns = ["critical_temp", "material"]
    symbol_data = symbol_data.drop(columns=symbol_drop_columns)

    subproblem_b_targets = symbol_data.copy()

    property_symbol_data = pd.concat([property_data, symbol_data], axis=1)

    # Model training

    mlp_model_2 = mlp(property_symbol_data, 50, seed=seed)
    rfr_model_2 = rfr(property_symbol_data, seed=seed)
    xgb_model_2 = xgb(property_symbol_data, 100, 0.1, 5, seed=seed)

    ml_models_2 = [mlp_model_2, rfr_model_2, xgb_model_2]
    
    for model in ml_models_2:
        model.process_data(subproblem_b_targets)
        model.train_model()
        model.test_prediction()
        model.classify_model_performance()
        print(model.get_statistics())

        model.create_graph(save_fig=save_files) # Saves graph if save_fig is True
        if save_files: # Savees stats if save_files is True
            model.save_statistics(model.name) 

if __name__ == "__main__":
    save_files = len(sys.argv) > 1 and sys.argv[1] is True
    main(save_files)

    plt.show()
