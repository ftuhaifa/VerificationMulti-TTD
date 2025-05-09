# -*- coding: utf-8 -*-

"""
Random Forest with SMOTEEvaluation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the data
main_data = pd.read_csv("LungCancer25.csv")
verified_data = pd.read_csv("encoded_merged_data03.csv")

# Define features and target
features = ['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM AJCC']
target = 'Survival years'

X = main_data[features]
y = main_data[target]

# Train-test split before SMOTE to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Prepare verification data
X_verify = verified_data[features]
y_verify = verified_data[target]

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train_smote.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))
y_verify_encoded = encoder.transform(y_verify.values.reshape(-1, 1))

# Define and train RandomForestClassifier
rf_model = RandomForestClassifier(max_depth=20, min_samples_leaf=1,
                                  min_samples_split=2, n_estimators=200,
                                  random_state=42)
rf_model.fit(X_train_smote, y_train_smote)

# Predict on testing data
y_test_pred = rf_model.predict(X_test)
y_test_prob = rf_model.predict_proba(X_test)

# Predict on verification data
y_verify_pred = rf_model.predict(X_verify)
y_verify_prob = rf_model.predict_proba(X_verify)

# Print unique classes and shape of probability predictions
print("Unique classes in training:", np.unique(y_train_smote))
print("Shape of y_verify_prob:", y_verify_prob.shape)

# Test set metrics
auc = roc_auc_score(y_test, y_test_prob, average='weighted', multi_class='ovr')
accuracy = accuracy_score(y_test, y_test_pred)
report = classification_report(y_test, y_test_pred, output_dict=True)
precision_general = report['weighted avg']['precision']
recall_general = report['weighted avg']['recall']
f1_general = report['weighted avg']['f1-score']

print('Accuracy:', accuracy)
print('General Precision:', precision_general)
print('General Recall:', recall_general)
print('General F1-Score:', f1_general)
print('Average AUC:', auc)

print('***********************************************************************')

# Verification set metrics
accuracy_v = accuracy_score(y_verify, y_verify_pred)
report_v = classification_report(y_verify, y_verify_pred, output_dict=True)
precision_general_v = report_v['weighted avg']['precision']
recall_general_v = report_v['weighted avg']['recall']
f1_general_v = report_v['weighted avg']['f1-score']

print('Accuracy Varification:', accuracy_v)
print('General Precision Varification:', precision_general_v)
print('General Recall Varification:', recall_general_v)
print('General F1-Score Varification:', f1_general_v)

print('***********************************************************************')

# Distribution plots
df = main_data
verification_df = verified_data

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

# Percentage distribution
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

def plot_feature_distribution(feature, ax_index):
    temp_train = df[feature].value_counts(normalize=True).reset_index()
    temp_train.columns = ['Category', 'Percentage']
    temp_train['Dataset'] = 'Training'
    
    temp_verify = verification_df[feature].value_counts(normalize=True).reset_index()
    temp_verify.columns = ['Category', 'Percentage']
    temp_verify['Dataset'] = 'Verification'
    
    combined = pd.concat([temp_train, temp_verify])
    
    sns.barplot(x='Category', y='Percentage', hue='Dataset', data=combined, ax=axes[ax_index])
    axes[ax_index].set_title(f'Percentage Distribution of {feature}')
    axes[ax_index].set_ylim(0, 1)
    axes[ax_index].set_ylabel('Percentage')
    axes[ax_index].set_xlabel(feature)

for i, feature in enumerate(features):
    plot_feature_distribution(feature, i)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()

# Distribution comparison
plt.figure(figsize=(18, 12))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    if X_train[feature].dtype in ['int64', 'float64']:
        sns.histplot(X_train[feature], label='Training', color='blue', kde=False, bins=30, stat='percent', alpha=0.6)
        sns.histplot(X_verify[feature], label='Verification', color='cyan', kde=False, bins=30, stat='percent', alpha=0.6)
    else:
        train_pct = X_train[feature].value_counts(normalize=True) * 100
        verification_pct = X_verify[feature].value_counts(normalize=True) * 100
        train_df = train_pct.reset_index().rename(columns={'index': feature, feature: 'Percentage'})
        verification_df_plot = verification_pct.reset_index().rename(columns={'index': feature, feature: 'Percentage'})
        sns.barplot(x=train_df[feature], y=train_df['Percentage'], label='Training', color='blue', alpha=0.6)
        sns.barplot(x=verification_df_plot[feature], y=verification_df_plot['Percentage'], label='Verification', color='orange', alpha=0.6)
    plt.title(f'Percentage Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Percentage')
    plt.legend()
plt.tight_layout()
plt.show()

# Final evaluation
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("Testing F1 Score:", f1_score(y_test, y_test_pred, average='macro'))
print("Testing ROC AUC Score:", roc_auc_score(y_test_encoded, y_test_prob, multi_class="ovr"))

print("Verification Accuracy:", accuracy_score(y_verify_encoded.argmax(axis=1), y_verify_pred))
print("Verification F1 Score:", f1_score(y_verify_encoded.argmax(axis=1), y_verify_pred, average='macro'))

print("\nClassification Report for Test Data:\n", classification_report(y_test_encoded.argmax(axis=1), y_test_pred))
print("\nClassification Report for Verification Data:\n", classification_report(y_verify_encoded.argmax(axis=1), y_verify_pred))
