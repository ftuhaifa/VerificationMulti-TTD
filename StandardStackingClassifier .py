# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
StandardStackingClassifier 
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
main_data = pd.read_csv("LungCancer25.csv")
verified_data = pd.read_csv("encoded_merged_data03.csv")

# Define features and target
features = ['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM AJCC']
target = 'Survival years'

# Define imputer and apply it
imputer = SimpleImputer(strategy='mean')
main_data[features] = imputer.fit_transform(main_data[features])
verified_data[features] = imputer.transform(verified_data[features])

# Split data (without SMOTE)
X_train, X_test, y_train, y_test = train_test_split(main_data[features],
                                                    main_data[target], test_size=0.3,
                                                    random_state=42)

X_verify = verified_data[features]
y_verify = verified_data[target]

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))
y_verify_encoded = encoder.transform(y_verify.values.reshape(-1, 1))

# Define base models
base_models = [
    ('rfc', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gbc', GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ('mlp', MLPClassifier(alpha=0.01, max_iter=200, random_state=42))
]

# Meta-classifier
meta_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Stacking classifier
rf_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_classifier,
    passthrough=True,
    stack_method='predict_proba',
    cv=StratifiedKFold(n_splits=5),
    verbose=1
)

# Fit model
rf_model.fit(X_train, y_train_encoded.argmax(axis=1))

# Predict on test and verification sets
y_test_pred = rf_model.predict(X_test)
y_test_prob = rf_model.predict_proba(X_test)
y_verify_pred = rf_model.predict(X_verify)
y_verify_prob = rf_model.predict_proba(X_verify)

# Print classes and prob shape
print("Unique classes in training:", np.unique(y_train))
print("Shape of y_verify_prob:", y_verify_prob.shape)

# Evaluate on test set
auc = roc_auc_score(y_test, y_test_prob, average='weighted', multi_class='ovr')
accuracy = accuracy_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred, output_dict=True)
print('Accuracy:', accuracy)
print('General Precision:', report['weighted avg']['precision'])
print('General Recall:', report['weighted avg']['recall'])
print('General F1-Score:', report['weighted avg']['f1-score'])
print('Average AUC:', auc)

print('***********************************************************************')

# Evaluate on verification set
accuracy = accuracy_score(y_verify, y_verify_pred)
report = classification_report(y_verify, y_verify_pred, output_dict=True)
print('Accuracy Varificaton:', accuracy)
print('General Precision Varification:', report['weighted avg']['precision'])
print('General Recall Varification:', report['weighted avg']['recall'])
print('General F1-Score Varification:', report['weighted avg']['f1-score'])

print('***********************************************************************')

# Plot distributions
df = pd.read_csv("LungCancer25.csv")
verification_df = pd.read_csv("encoded_merged_data03.csv")

fig, axes = plt.subplots(nrows=len(features) + 1, ncols=2, figsize=(15, 5 * (len(features) + 1)), sharex='col')
sns.countplot(x=df[target], ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title('Distribution of Survival Years in Main Data')
sns.countplot(x=verification_df[target], ax=axes[0, 1], palette="viridis")
axes[0, 1].set_title('Distribution of Survival Years in Verification Data')

for i, feature in enumerate(features, start=1):
    sns.histplot(df[feature], bins=20, kde=True, ax=axes[i, 0], color='blue', stat="density")
    axes[i, 0].set_title(f'Distribution of {feature} in Main Data')
    sns.histplot(verification_df[feature], bins=20, kde=True, ax=axes[i, 1], color='green', stat="density")
    axes[i, 1].set_title(f'Distribution of {feature} in Verification Data')

plt.tight_layout()
plt.show()

# Percentage bar plots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, feature in enumerate(features[:-1]):
    temp_train = df[feature].value_counts(normalize=True).reset_index()
    temp_train.columns = ['Category', 'Percentage']
    temp_train['Dataset'] = 'Training'

    temp_verify = verification_df[feature].value_counts(normalize=True).reset_index()
    temp_verify.columns = ['Category', 'Percentage']
    temp_verify['Dataset'] = 'Verification'

    combined = pd.concat([temp_train, temp_verify])
    sns.barplot(x='Category', y='Percentage', hue='Dataset', data=combined, ax=axes[i])
    axes[i].set_title(f'Percentage Distribution of {feature}')
    axes[i].set_ylim(0, 1)
    axes[i].set_ylabel('Percentage')
    axes[i].set_xlabel(feature)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()

# Final evaluation
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("Testing F1 Score:", f1_score(y_test, y_test_pred, average='macro'))
print("Testing ROC AUC Score:", roc_auc_score(y_test_encoded, y_test_prob, multi_class="ovr"))

print("Verification Accuracy:", accuracy_score(y_verify_encoded.argmax(axis=1), y_verify_pred))
print("Verification F1 Score:", f1_score(y_verify_encoded.argmax(axis=1), y_verify_pred, average='macro'))

print("\nClassification Report for Test Data:\n", classification_report(y_test_encoded.argmax(axis=1), y_test_pred))
print("\nClassification Report for Verification Data:\n", classification_report(y_verify_encoded.argmax(axis=1), y_verify_pred))
