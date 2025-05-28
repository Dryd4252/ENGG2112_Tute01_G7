import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import randint, uniform
from scipy.optimize import differential_evolution

from ml_models.mlp import MlpModel as mlp
from ml_models.rfr import RfrModel as rfr
from ml_models.xgb import XgbModel as xgb

def main(save_files):
    seed = None

    ### Sub Problem A
    
    # Data processing
    property_data = pd.read_csv("train.csv")

    mlp_model_1 = mlp(property_data, 50, seed=seed)
    rfr_model_1 = rfr(property_data, seed=seed)
    xgb_model_1 = xgb(property_data, 100, 0.1, 5, seed=seed)

    ml_models_1 = [mlp_model_1, rfr_model_1, xgb_model_1]

    results_1 = []

    print("Subproblem A")
    for model in ml_models_1:
        model.process_data(["critical_temp"])
        model.train_model()
        model.test_prediction()
        model.classify_model_performance()
        results_1.append(model.get_statistics())
        print(f"{model.__class__.__name__}: ")
        print(model.get_statistics())
        model.create_graph(save_fig=save_files) # Saves graph if save_fig is True

        if save_files: # Savees stats if save_files is True
            model.save_statistics(model.name) 
    
    # Create plot with subplots comparing the model results
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Subproblem A: Model Performance Comparison')
    metrics = ['mse', 'rmse', 'mae', 'r2']
    colors = ["#ffab40", "#85d5e6", "#0097a7"]
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        axs[i].bar([model.name for model in ml_models_1], res:=[result[i] for result in results_1], color=colors)
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

    results_2 = []

    print("\nSubproblem B")
    for model in ml_models_2:
        model.process_data(subproblem_b_targets)
        model.train_model()
        model.test_prediction()
        model.classify_model_performance()
        results_2.append(model.get_statistics())
        print(f"{model.__class__.__name__}: ")
        print(model.get_statistics())
        if save_files: # Savees stats if save_files is True
            model.save_statistics(model.name) 

    # Create plot with subplots comparing the model results
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Subproblem B: Model Performance Comparison')
    metrics = ['mse', 'rmse', 'mae', 'r2']
    colors = ["#ffab40", "#85d5e6", "#0097a7"]
    axs = axs.flatten()
    for i, metric in enumerate(metrics):
        axs[i].bar([model.name for model in ml_models_1], res:=[result[i] for result in results_2], color=colors)
        axs[i].set_title(metric.upper())
        padding = (max(res) - min(res)) * 0.2
        axs[i].set_ylim([min(res) - padding, max(res) + padding])


    ### Sub Problem C
    
    # Obtain property labels and ranges
    global property_labels
    property_labels = []
    property_range_list = []

    for label, data in property_data.drop(columns="critical_temp").items():
        property_range_list.append((data.min(), data.max()))
        property_labels.append(label)

    # Find maximum temperature predicted of each model
    model_combination_max_Tc = {}
    for model in ml_models_1:
        model_name = model.__class__.__name__

        global global_model
        global_model = model

        result = differential_evolution(
            global_model_make_prediction, 
            property_range_list,
            workers=-1,
            disp=True)
        
        if result.success == False:
            print("{model_name}: failed optomise")

        optimal_properties = result.x
        max_predicted_temperature = -result.fun
        model_combination_max_Tc[model_name] = (optimal_properties, max_predicted_temperature)

    for model, properties_temperature in model_combination_max_Tc.items():
        model_name = model.__class__.__name__
        print(f"For {model_name}: ") 
        print(f"Optimal properties: {properties_temperature[0]}")
        print(f"Predicted critical temperature: {properties_temperature[1]}")

# Global function definition for model make_prediction
# For compatability with differential_evolution and allow for multi threading
def global_model_make_prediction(x):
    return -global_model.make_prediction(pd.DataFrame([x], columns=property_labels))[0]

if __name__ == "__main__":
    save_files = len(sys.argv) > 1 and sys.argv[1] is True
    main(save_files)

    plt.show()
