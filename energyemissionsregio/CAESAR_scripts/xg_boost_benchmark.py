import os
import pandas as pd

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

corr_threshold = 0.5
var_to_impute = "de_residential_building_living_area"

print(
    f"running XGB benchmark with corr: {corr_threshold}, var_to_impute: {var_to_impute}==============================================="
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
model = XGBRegressor()


model.fit(X_train, y_train)

y_pred = model.predict(X_train)

rmse = mean_squared_error(y_train, y_pred, squared=False)
mae = mean_absolute_error(y_train, y_pred)
r2 = r2_score(y_train, y_pred)


print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)


# ===============================================================
# Testing

y_pred = model.predict(X_valid)

rmse = mean_squared_error(y_valid, y_pred, squared=False)
mae = mean_absolute_error(y_valid, y_pred)
r2 = r2_score(y_valid, y_pred)

print("RMSE score on validation:", rmse)
print("MAE score on validation:", mae)
print("r2 score on validation:", r2)
