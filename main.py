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


def rfr_test_optomiser(property_data: pd.DataFrame, target_labels: list[str], seed: int):
    param_grid = {
        'n_estimators': randint(100, 300),
        'max_depth': [None] +  list(range(10, 51, 10)),
        'min_samples_split': randint(2, 5),
        'min_samples_leaf': randint(1, 2),
        'max_features': ['sqrt', 'log2']
    }

    rfr_model_t = rfr(property_data, seed=seed)
    rfr_model_t.process_data(target_labels)
    rfr_model_t.optimise_model(param_grid, n_iter=5)
    rfr_model_t.test_prediction()
    rfr_model_t.classify_model_performance()

    print(rfr_model_t.get_statistics())
    print(rfr_model_t.get_params())

def xgb_optimiser(property_data: pd.DataFrame, seed: int):
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
    xgb_model_t.optimise_model(param_grid, n_iter=10)
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

    # print("Calling xgb_optimiser() for the first time to train model")
    # xgb_model_t = xgb_optimiser(property_data, seed)
    # plot_property_importance(xgb_model_t, property_data
    # plot_top_properties_boxplots(xgb_model_t, property_data)

    # for i in range(2):
    #     xgb_model_t = xgb_optimiser(property_data, seed)

    #     if save_files: # Savees stats if save_files is True
    #         xgb_model_t.save_statistics(xgb_model_t.name) 

    # rfr_test_optomiser(property_data, seed)

    mlp_model_1 = mlp(property_data, 50, seed=seed)
    rfr_model_1 = rfr(property_data, seed=seed)
    xgb_model_1 = xgb(property_data, 100, 0.1, 5, seed=seed)

    ml_models_1 = [mlp_model_1, rfr_model_1, xgb_model_1]

    results = []
    for model in ml_models_1:
        model.process_data(["critical_temp"])
        model.train_model()
        model.test_prediction()
        model.classify_model_performance()
        results.append(model.get_statistics())
        print(model.get_statistics())
        # model.create_graph(save_fig=save_files) # Saves graph if save_fig is True

        if save_files: # Savees stats if save_files is True
            model.save_statistics(model.name) 
    
    # Create plot with subplots comparing the model results
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Model Performance Comparison')
    metrics = ['mse', 'rmse', 'mae', 'r2']
    colors = ["#ffab40", "#85d5e6", "#0097a7"]
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        axs[i].bar([model.name for model in ml_models_1], res:=[result[i] for result in results], color=colors)
        axs[i].set_title(metric.upper())
        padding = (max(res) - min(res)) * 0.2
        axs[i].set_ylim([min(res) - padding, max(res) + padding])

    ### Sub Problem B

    # Data processing
    symbol_data = pd.read_csv("unique_m.csv")
    symbol_drop_columns = ["critical_temp", "material"]
    symbol_data = symbol_data.drop(columns=symbol_drop_columns)

    subproblem_b_targets = symbol_data.columns
    property_symbol_data = pd.concat([property_data, symbol_data], axis=1).drop(columns="critical_temp")

    # Model training

    mlp_model_2 = mlp(property_symbol_data, 50, seed=seed)
    rfr_model_2 = rfr(property_symbol_data, 
        seed=seed, 
        max_depth=40, 
        max_features="log2", 
        min_samples_leaf=1,
        min_samples_split=2,
        n_estimators=212)
    xgb_model_2 = xgb(property_symbol_data, 100, 0.1, 5, seed=seed)

    ml_models_2 = [mlp_model_2, rfr_model_2, xgb_model_2]
    
    for model in ml_models_2:
        model.process_data(subproblem_b_targets)
        model.train_model()
        model.test_prediction()
        model.classify_model_performance()
        print(f"{model.__class__.__name__}: ")
        print(model.get_statistics())
        if save_files: # Savees stats if save_files is True
            model.save_statistics(model.name) 
    
    # sample_material_property = [5,92.729214,58.5184161428571,73.1327872225065,36.3966020291995,1.44930919335685,1.05775512271911,122.90607,36.161939,47.0946331703134,53.9798696513451,766.44,1010.61285714286,720.605510513725,938.745412527433,1.54414454326973,0.807078214938731,810.6,743.164285714286,290.183029138508,354.963511171592,161.2,104.971428571429,141.465214777999,84.3701669575628,1.50832754035259,1.2041147982326,205,50.5714285714286,67.321319060161,68.0088169554027,5821.4858,3021.01657142857,1237.09508033858,54.0957182556368,1.31444218462105,0.914802177066343,10488.571,1667.38342857143,3767.40317577062,3632.64918471043,90.89,112.316428571429,69.8333146094209,101.166397739874,1.42799655342352,0.83866646563365,127.05,81.2078571428572,49.4381674417651,41.6676207979191,7.7844,3.79685714285714,4.40379049753476,1.03525111582814,1.37497728009085,1.07309384625263,12.878,1.59571428571429,4.47336265464807,4.60300005985449,172.205316,61.3723314285714,16.0642278788044,0.619734632330547,0.847404163195705,0.567706107876637,429.97342,51.4133828571429,198.554600255545,139.630922368904,2,2.25714285714286,1.8881750225898,2.21067940870655,1.55711309805765,1.04722136819323,2,1.12857142857143,0.632455532033676,0.468606270481621]
    
    # for model in ml_models_2:
    #     expected_composite = model.make_prediction(pd.DataFrame([x], columns=property_data.drop(columns="critical_temp").columns))
    #     print(expected_composite)


if __name__ == "__main__":
    save_files = len(sys.argv) > 1 and sys.argv[1] is True
    main(save_files)

    plt.show()
