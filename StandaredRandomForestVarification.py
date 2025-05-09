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
#smote = SMOTE(random_state=42)
#X, y = smote.fit_resample(X, y)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Prepare verification data
X_verify = verified_data[features]
y_verify = verified_data[target]

# One-hot encode the target variable
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))  # Encode y_test
y_verify_encoded = encoder.transform(y_verify.values.reshape(-1, 1))

# Define and train RandomForestClassifier
rf_model = RandomForestClassifier(max_depth=20, min_samples_leaf=1,
                                  min_samples_split=2, n_estimators=200,
                                  random_state=42)
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


# Custom colors
custom_palette = ["blue", "cyan"]


# Helper function to plot a feature
def plot_feature_distribution(feature, ax_index):
    # Combine the data for both datasets
    temp_train = df[feature].value_counts(normalize=True).reset_index()
    temp_train.columns = ['Category', 'Percentage']
    temp_train['Dataset'] = 'Training'
    
    temp_verify = verification_df[feature].value_counts(normalize=True).reset_index()
    temp_verify.columns = ['Category', 'Percentage']
    temp_verify['Dataset'] = 'Verification'
    
    combined = pd.concat([temp_train, temp_verify])
    
    # Plot
    sns.barplot(x='Category', y='Percentage', hue='Dataset', data=combined, ax=axes[ax_index])
    axes[ax_index].set_title(f'Percentage Distribution of {feature}')
    axes[ax_index].set_ylim(0, 1)  # Adjust as necessary to match scale in your example
    axes[ax_index].set_ylabel('Percentage')
    axes[ax_index].set_xlabel(feature)

# Plot each feature
for i, feature in enumerate(features):
    plot_feature_distribution(feature, i)

# Remove unused subplots if there are any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
plt.show()










# Set up subplots to visualize the percentage distribution of each feature
plt.figure(figsize=(18, 12))

for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    
    # Check if the feature is numerical or categorical
    if X_train[feature].dtype in ['int64', 'float64']:
        # Plot normalized histogram for numerical features (percentages)
        sns.histplot(X_train[feature], label='Training', color='blue', kde=False, bins=30, 
                     stat='percent', alpha=0.6)
        sns.histplot(X_verify[feature], label='Verification', color='cyan', kde=False, 
                     bins=30, stat='percent', alpha=0.6)
        plt.title(f'Percentage Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Percentage')
    else:
        # Normalize the counts for categorical features
        train_pct = X_train[feature].value_counts(normalize=True) * 100
        verification_pct = X_verify[feature].value_counts(normalize=True) * 100
        
        # Convert to dataframes for easier plotting
        train_df = train_pct.reset_index().rename(columns={'index': feature, feature: 'Percentage'})
        verification_df = verification_pct.reset_index().rename(columns={'index': feature, feature: 'Percentage'})
        
        # Plot as bar plot for categorical features
        sns.barplot(x=train_df[feature], y=train_df['Percentage'], label='Training', color='blue', alpha=0.6)
        sns.barplot(x=verification_df[feature], y=verification_df['Percentage'], label='Verification', color='orange', alpha=0.6)
        
        plt.title(f'Percentage Distribution of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Percentage')

    plt.legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()




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
