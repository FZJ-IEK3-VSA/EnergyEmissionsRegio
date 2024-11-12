import os
import sys
import pickle
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    make_scorer,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Retrieve the array index passed from the Slurm script
corr_threshold = float(sys.argv[1])

var_to_impute = "de_non_residential_building_living_area"

print(
    f"running XGB with corr: {corr_threshold}, var_to_impute: {var_to_impute}==============================================="
)

# ===============================================================
final_df = pd.read_csv(
    os.path.join("data", f"{var_to_impute}_{corr_threshold}corr.csv")
)


X = final_df[~final_df[var_to_impute].isna()].drop(columns=[var_to_impute]).copy()
y = final_df[~final_df[var_to_impute].isna()][[var_to_impute]].copy()

print(f"Total records = {len(final_df)}")
print(f"Records with no NAs = {len(X)}")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=42
)

print(f"Number of train records = {len(X_train)}")
print(f"Number of test records = {len(X_valid)}")


# ================================================================
param_grid = {
    "tree_method": ["gpu_hist"],
    "device": ["cuda"],
    "objective": ["count:poisson"],
    "base_score": [y_train[var_to_impute].mean()],
    "n_estimators": [100, 300, 500, 700],
    "learning_rate": [0, 0.3, 0.5, 0.7, 1],
    "max_depth": [0, 2, 4, 6, 8],
}
scoring = {
    "RMSE": make_scorer(
        root_mean_squared_error, squared=False, greater_is_better=False
    ),
    "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
    "R2": "r2",  # You can directly use 'r2' as it's a built-in scoring function
}

# Initialize GridSearchCV
grid_search = GridSearchCV(
    XGBRegressor(),
    param_grid,
    cv=5,
    scoring=scoring,
    refit="RMSE",
    verbose=2,
)

grid_search.fit(X_train, y_train)

# Access the results
results = grid_search.cv_results_

# Get the index of the best parameters (based on RMSE because refit='RMSE')
best_index = grid_search.best_index_

# Extract and print the best scores for all metrics using the best index
best_rmse_score = results["mean_test_RMSE"][best_index]
best_mae_score = results["mean_test_MAE"][best_index]
best_r2_score = results["mean_test_R2"][best_index]

print("Best parameters:", grid_search.best_params_)
print("Best scores for the chosen parameters:")
print("Best RMSE:", best_rmse_score)
print("Best MAE:", best_mae_score)
print("Best R2:", best_r2_score)


# ===============================================================
# Testing

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_valid)

rmse = root_mean_squared_error(y_valid, y_pred, squared=False)
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print("RMSE score on validation:", rmse)
print("MAE score on validation:", mae)
print("r2 score on validation:", r2)

# ===============================================================
# Save best model to a file
with open(
    os.path.join("data", "models", f"{var_to_impute}_xgb_{corr_threshold}corr.pkl"),
    "wb",
) as file:
    pickle.dump(best_model, file)
