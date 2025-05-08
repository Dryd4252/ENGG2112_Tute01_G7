import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load data
data = pd.read_csv("train.csv")
y = data["critical_temp"]
X = data.drop(columns=["critical_temp"])

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69420)

# 3. Define model and parameter grid
rf = RandomForestRegressor(random_state=69420)

param_grid = {
    'n_estimators': randint(100, 300),
    'max_depth': [None] + randint(10, 20),
    'min_samples_split': randint(2, 5),
    'min_samples_leaf': randint(1, 2),
    'max_features': ['sqrt', 'log2']
}

# 4. Set up GridSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_grid=param_grid,
    n_iter=20,  # only try 20 random combinations
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=69420,
    verbose=1
)

# 5. Run grid search
random_search.fit(X_train, y_train)

# 6. Evaluate the best model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Best Parameters:", random_search.best_params_)
print("Test MSE:", mse)
