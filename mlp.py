import pandas as pd

all_data = pd.read_csv("train.csv")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix,roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm



csv_file = "train.csv" # csv file to get data from
label = "critical_temp" # label we are trying to predict using features
seed = 1000 # seed for the random state, for repeatability


data = pd.read_csv(csv_file) # read data from csv file
data.dropna() # remove missing/nan values from the data set 


# get class features
X = data.drop(columns=[label])
# get class labels
y = data[label]
# ecode y class labels for mlp, not a necessary step but it works either way
# encoder = LabelEncoder()
# y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist() 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed) # This random_state sets the seed. Remove it to randomly split the data every time.




def mlp_model(size):
    progress_bar = tqdm(total=6, desc=f"Processing steps {size}")
    progress_bar.update(1)
    mlp_preprocessor = ColumnTransformer(
        transformers=[
            ("categorical",OneHotEncoder(handle_unknown='ignore'),categorical_features),
            ("scaler",StandardScaler(), numerical_features)
        ]
    )

    progress_bar.update(1)

    mlp_model = Pipeline(steps=[
        ('preprocessor',mlp_preprocessor),
        ('mlp', MLPRegressor(hidden_layer_sizes=(size, size),max_iter=10000, random_state=seed))
    ])
    progress_bar.update(1)


    mlp_model.fit(X_train, y_train)
    y_pred = mlp_model.predict(X_test) #predicts the values of y_test based on X_test

    # print("Predictions: ", y_pred[:5])
    # print("Actual: ", y_test[:5])
    progress_bar.update(1)


    # Predict using the model




    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    progress_bar.update(1)

    results_df = pd.DataFrame({
        "mse": [mse],
        "rmse": [rmse],
        "mae": [mae],
        "r2": [r2]
    })

    results_df.to_csv(f"mlp_results\\txt\{size,size} layers results")



    # Scatter plot of actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect prediction line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'MLP Regressor: Actual vs Predicted ({size})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"mlp_results\graphs\{size} layers model.png")
    progress_bar.update(1)

sizes = [50,60,70,80,90,100]
for size in sizes:
    mlp_model(size) 