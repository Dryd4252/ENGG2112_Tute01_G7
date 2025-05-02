## Import All Packages

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


## Set File Information

csv_file = "train.csv" # csv file to get data from
label = "critical_temp" # label we are trying to predict using features
seed = 1000 # seed for the random state, for repeatability



sizes = [50,60,70,80,90,100]
for size in sizes:
    mlp_model(size) 