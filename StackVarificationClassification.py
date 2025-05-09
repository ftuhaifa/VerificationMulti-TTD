# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:47:53 2024

@author: ftuha
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:30:57 2024
@author: ftuha
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler  # Import RandomUnderSampler
from scipy import stats
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score

# Load and preprocess the data
main_data = pd.read_csv("LungCancer25.csv")
verified_data = pd.read_csv("encoded_merged_data03.csv")

# Define features and target
features = ['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM AJCC']
target = 'Survival years'

# Split main data into train and test

X = main_data[features]
y = main_data[target]


# Balance the classes using over-sampling
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

from sklearn.impute import SimpleImputer

# Define imputer (you can change the strategy to 'median' or 'most_frequent' if that makes more sense for your data)
imputer = SimpleImputer(strategy='mean')

# Apply imputation to your features
main_data[features] = imputer.fit_transform(main_data[features])
verified_data[features] = imputer.transform(verified_data[features])

# Now split your data
X_train, X_test, y_train, y_test = train_test_split(main_data[features],
                                                    main_data[target], test_size=0.3,
                                                    random_state=42)
# Apply SMOTE to balance the training dataset
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

X_verify = verified_data[features]
y_verify = verified_data[target]

# The rest of your setup follows
# Make sure you've checked and handled all columns that could have missing data in both training and verification datasets


# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))  # Encode y_test
y_verify_encoded = encoder.transform(y_verify.values.reshape(-1, 1))



# Define the base models
base_models = [
    ('rfc', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gbc', GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ('mlp', MLPClassifier(alpha=0.01, max_iter=200, random_state=42))
]

# Define the meta-classifier
meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Define the StackingClassifier
rf_model = StackingClassifier(
    estimators=base_models,  # This is how you set the base models in sklearn's StackingClassifier
    final_estimator=meta_classifier,  # This is the meta-classifier
    passthrough=True,  # Use features in secondary (meta) model
    stack_method='predict_proba',  # Use the probability outputs of the base models for the meta model
    cv=StratifiedKFold(n_splits=5),  # Cross-validation strategy
    verbose=1  # Verbose output
)





















rf_model.fit(X_train, y_train_encoded.argmax(axis=1))

# Predict on testing data
y_test_pred = rf_model.predict(X_test)
y_test_prob = rf_model.predict_proba(X_test)

# Predict on verification data
y_verify_pred = rf_model.predict(X_verify)
y_verify_prob = rf_model.predict_proba(X_verify)





# Print unique classes and shape of probability predictions
print("Unique classes in training:", np.unique(y_train))
print("Shape of y_verify_prob:", y_verify_prob.shape)





# Calculate ROC AUC for multi-class classification
auc = roc_auc_score(y_test, y_test_prob , average='weighted', multi_class='ovr')

# Compute and print overall metrics
accuracy = accuracy_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred, output_dict=True)

# General scores
precision_general = report['weighted avg']['precision']
recall_general = report['weighted avg']['recall']
f1_general = report['weighted avg']['f1-score']

print('Accuracy:', accuracy)
print('General Precision:', precision_general)
print('General Recall:', recall_general)
print('General F1-Score:', f1_general)
print('Average AUC:', auc)






print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')







# Compute and print overall metrics
accuracy = accuracy_score(y_verify, y_verify_pred)
report = classification_report(y_verify, y_verify_pred, output_dict=True)

# General scores
precision_general = report['weighted avg']['precision']
recall_general = report['weighted avg']['recall']
f1_general = report['weighted avg']['f1-score']

print('Accuracy Varificaton:', accuracy)
print('General Precision Varification:', precision_general)
print('General Recall Varification:', recall_general)
print('General F1-Score Varification:', f1_general)

# Calculate ROC AUC for multi-class classification
#auc = roc_auc_score(y_verify, y_verify_prob , average='weighted', multi_class='ovo')
#print('Average AUC Varification:', auc)


print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')
print('***********************************************************************')


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("LungCancer25.csv")
verification_df = pd.read_csv("encoded_merged_data03.csv")

# Define features
features = ['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM AJCC']
target = 'Survival years'

# Setup the matplotlib figure and grid
n_features = len(features)
fig, axes = plt.subplots(nrows=n_features + 1, ncols=2, figsize=(15, 5 * (n_features + 1)), sharex='col')

# Plotting target variable distribution
sns.countplot(x=df[target], ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title('Distribution of Survival Years in Main Data')
axes[0, 0].set_xlabel('')
axes[0, 0].set_ylabel('Count')

sns.countplot(x=verification_df[target], ax=axes[0, 1], palette="viridis")
axes[0, 1].set_title('Distribution of Survival Years in Verification Data')
axes[0, 1].set_xlabel('')
axes[0, 1].set_ylabel('Count')

# Plotting distributions for each feature
for i, feature in enumerate(features, start=1):
    sns.histplot(df[feature], bins=20, kde=True, ax=axes[i, 0], color='blue', stat="density")
    axes[i, 0].set_title(f'Distribution of {feature} in Main Data')
    axes[i, 0].set_xlabel(feature)
    axes[i, 0].set_ylabel('Density')

    sns.histplot(verification_df[feature], bins=20, kde=True, ax=axes[i, 1], color='green', stat="density")
    axes[i, 1].set_title(f'Distribution of {feature} in Verification Data')
    axes[i, 1].set_xlabel(feature)
    axes[i, 1].set_ylabel('Density')

# Adjusting layout
plt.tight_layout()
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("LungCancer25.csv")
verification_df = pd.read_csv("encoded_merged_data03.csv")

# Define features
features = ['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology']
target = 'Survival years'

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

# Flatten the axes array for easy iteration
axes = axes.flatten()






# Evaluation metrics for test data
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("Testing F1 Score:", f1_score(y_test, y_test_pred, average='macro'))
print("Testing ROC AUC Score:", roc_auc_score(y_test_encoded, y_test_prob, multi_class="ovr"))

# Evaluation metrics for verification data
print("Verification Accuracy:", accuracy_score(y_verify_encoded.argmax(axis=1), y_verify_pred))
print("Verification F1 Score:", f1_score(y_verify_encoded.argmax(axis=1), y_verify_pred, average='macro'))
#print("Verification ROC AUC Score:", roc_auc_score(y_verify_encoded, y_verify_prob, multi_class="ovr"))

# Classification reports for more detailed analysis
print("\nClassification Report for Test Data:\n", classification_report(y_test_encoded.argmax(axis=1), y_test_pred))
print("\nClassification Report for Verification Data:\n", classification_report(y_verify_encoded.argmax(axis=1), y_verify_pred))
