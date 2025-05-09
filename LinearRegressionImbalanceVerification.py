# -*- coding: utf-8 -*-
"""

"""

#linear regression normal Varification


from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import pandas as pd

# Load and preprocess the data
main_data = pd.read_csv("LungCancer25.csv")
verified_data = pd.read_csv("encoded_merged_data03.csv")

# Define features and target
features = ['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM AJCC']
target = 'Survival years'




# Split data into features and target
#X = df.drop("Survival years", axis=1)
#y = df["Survival years"]

X = main_data[features]
y = main_data[target]

# Normalize features using min-max normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters and their range of values
param_grid = {'fit_intercept': [True, False],
              'copy_X': [True, False]}


#To add a grid search for hyperparameters, we can use the GridSearchCV
#function from scikit-learn. First, we define the hyperparameters we want to tune
# and the range of values to try. Then, we create a GridSearchCV object and
# fit it to the training data. Finally, we use the best hyperparameters to
# train a new model and evaluate its performance on the test set.


# Create a GridSearchCV object
grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Train a new model using the best hyperparameters
best_lr = LinearRegression(**grid_search.best_params_)
best_lr.fit(X_train, y_train)

# Evaluate the new model on the test set
y_pred = best_lr.predict(X_test)

#y_pred = y_pred.astype(np.int)
#y_pred = np.where(y_pred > 0.5, 1, 0)
#y_pred = np.round(y_pred).astype(int)


#y_pred = np.round(y_pred).astype(int) - 1
#y_pred = y_pred + 1


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

#adding standard deviation
std_dev = np.std(y_pred)
#Calculate standard deviation of residuals
residuals = y_test - y_pred
std_residuals = np.std(residuals)

#Calculate mean of the target variable
y_mean = np.mean(y_test)

#Calculate weighting factor for custom ensemble
w = np.abs(residuals / (y_test - y_mean))

print("***** TRain Data ******")

print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')
print(f'Standard deviation: {std_dev}')
print(f'standard deviation of residuals: {std_residuals}')
print(f'mean of the target variable: {y_mean}')
print(f'Weighting Factor: {w}')







#Validation
#target ='Continuous Years from Diagnosis to Death'
X_verify = verified_data[features]
y_verify = verified_data[target]


from sklearn.impute import SimpleImputer

# Imputer for handling missing values
imputer = SimpleImputer(strategy='mean')

# Preprocessing the verification data
X_verify_preprocessed = imputer.fit_transform(X_verify)

# Predict on the preprocessed verification data
y_verify_pred = best_lr.predict(X_verify_preprocessed)


# Evaluate the new model on the test set
# Predict on verification data
#y_verify_pred = best_lr.predict(X_verify)
#y_verify_prob = best_lr.predict_proba(X_verify)



rmse = np.sqrt(mean_squared_error(y_verify, y_verify_pred))
r2 = r2_score(y_verify, y_verify_pred)

#adding standard deviation
std_dev = np.std(y_verify_pred)
#Calculate standard deviation of residuals
residuals = y_verify - y_verify_pred
std_residuals = np.std(residuals)

#Calculate mean of the target variable
y_mean = np.mean(y_verify)

#Calculate weighting factor for custom ensemble
w = np.abs(residuals / (y_verify - y_mean))



print("***** Varification Data ******")

print(f'RMSE: {rmse}')
print(f'R-squared: {r2}')
print(f'Standard deviation: {std_dev}')
print(f'standard deviation of residuals: {std_residuals}')
print(f'mean of the target variable: {y_mean}')
print(f'Weighting Factor: {w}')





















#*********************************************************************

#To add confidence intervals for the parameters,
# we can use the conf_int() method from the statsmodels library.
# This method returns the confidence intervals for each coefficient in
# the model.


# Fit a linear regression model using statsmodels
X_train_sm = sm.add_constant(X_train)
lr_sm = sm.OLS(y_train, X_train_sm).fit()

# Print the summary of the model
print(lr_sm.summary())

# Get the confidence intervals for the parameters
conf_int = lr_sm.conf_int(alpha=0.05)
conf_int.columns = ['Lower CI', 'Upper CI']
print(conf_int)
#*********************************************************************

#Print the standard deviation of residuals, mean, and weighting factor
y_pred_train = best_lr.predict(X_train)
residuals_train = y_train - y_pred_train
std_dev_resid_train = np.std(residuals_train)


print("################################################")
print("################################################")
print("################################################")
print("################################################")
print("################################################")
summary = lr_sm.summary2()
coef_table = summary.tables[1]
# Get the coefficient table as a dataframe
coef_table = lr_sm.summary2().tables[1]
coef_df = coef_table[['Coef.', 'Std.Err.', 't', 'P>|t|']]
print(coef_df)

# Create a plot of the coefficient estimates and their confidence intervals
fig, ax = plt.subplots()
ax.errorbar(x=coef_df['Coef.'], y=coef_df.index, xerr=1.96*coef_df['Std.Err.'],
            fmt='o', capsize=5, elinewidth=2)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_xlabel('Coefficient Estimate')
ax.set_ylabel('Variable')
ax.set_title('Interpreting p-values and coefficients')
plt.show()


import matplotlib.pyplot as plt

# assuming y_pred and y_actual are the predicted and actual survival times, respectively
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Survival Time (years)')
plt.ylabel('Predicted Survival Time (years)')
plt.title('Linear regression')
plt.show()