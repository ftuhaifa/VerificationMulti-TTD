# -*- coding: utf-8 -*-
"""
Linear Regression with SMOTE, Verification, Confidence Intervals, and Visualizations
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load and define
main_data = pd.read_csv("LungCancer25.csv")
verified_data = pd.read_csv("encoded_merged_data03.csv")

features = ['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM AJCC']
target = 'Survival years'

X = main_data[features]
y = main_data[target]

# MinMax scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Round target to nearest int for SMOTE
y_int = y.round().astype(int)

# Split before SMOTE to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_int, test_size=0.2, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Grid search for best LinearRegression params
param_grid = {'fit_intercept': [True, False], 'copy_X': [True, False]}
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5)
grid_search.fit(X_train_res, y_train_res)

print("Best hyperparameters:", grid_search.best_params_)

# Fit best model
best_lr = LinearRegression(**grid_search.best_params_)
best_lr.fit(X_train_res, y_train_res)

# Predict on test set
y_pred = best_lr.predict(X_test)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
std_dev = np.std(y_pred)
residuals = y_test - y_pred
std_residuals = np.std(residuals)
y_mean = np.mean(y_test)
w = np.abs(residuals / (y_test - y_mean))

print("***** Test Data ******")
print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')
print(f'Standard deviation: {std_dev}')
print(f'Std of residuals: {std_residuals}')
print(f'Mean target: {y_mean}')
print(f'Weighting Factor: {w}')

# Verification data
X_verify = verified_data[features]
y_verify = verified_data[target]
X_verify_preprocessed = SimpleImputer(strategy='mean').fit_transform(X_verify)
X_verify_scaled = scaler.transform(X_verify_preprocessed)
y_verify_pred = best_lr.predict(X_verify_scaled)

rmse_v = np.sqrt(mean_squared_error(y_verify, y_verify_pred))
r2_v = r2_score(y_verify, y_verify_pred)
std_dev_v = np.std(y_verify_pred)
residuals_v = y_verify - y_verify_pred
std_residuals_v = np.std(residuals_v)
y_mean_v = np.mean(y_verify)
w_v = np.abs(residuals_v / (y_verify - y_mean_v))

print("***** Verification Data ******")
print(f'RMSE: {rmse_v}')
print(f'R-squared: {r2_v}')
print(f'Standard deviation: {std_dev_v}')
print(f'Std of residuals: {std_residuals_v}')
print(f'Mean target: {y_mean_v}')
print(f'Weighting Factor: {w_v}')

# Confidence intervals using statsmodels
X_train_sm = sm.add_constant(X_train_res)
lr_sm = sm.OLS(y_train_res, X_train_sm).fit()
print(lr_sm.summary())

conf_int = lr_sm.conf_int(alpha=0.05)
conf_int.columns = ['Lower CI', 'Upper CI']
print(conf_int)

# Coefficient plot
summary = lr_sm.summary2()
coef_table = summary.tables[1]
coef_df = coef_table[['Coef.', 'Std.Err.', 't', 'P>|t|']]
fig, ax = plt.subplots()
ax.errorbar(x=coef_df['Coef.'], y=coef_df.index, xerr=1.96*coef_df['Std.Err.'], fmt='o', capsize=5)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel('Coefficient Estimate')
ax.set_ylabel('Variable')
ax.set_title('Coefficients and 95% CI')
plt.show()

# Scatter plot of prediction vs actual
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Survival Time (years)')
plt.ylabel('Predicted Survival Time (years)')
plt.title('Linear Regression - SMOTE Applied')
plt.grid(True)
plt.show()
