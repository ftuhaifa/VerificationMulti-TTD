# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 07:35:14 2024

@author: ftuha
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 14:56:37 2024

@author: faaa272
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, accuracy_score
from sklearn.model_selection import train_test_split
import random
import time
import matplotlib.pyplot as plt

start_time = time.time()

# Load the training dataset (LungCancer32.csv)
data_train = pd.read_csv('LungCancer32.csv')

X = data_train[['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM']]
y = data_train[['DX-bone', 'DX-brain', 'DX-liver']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load the verification dataset (LungCancer06.csv)
data_verification = pd.read_csv('LungCancer06.csv')

X_verification = data_verification[['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology', 'TNM']]
y_verification = data_verification[['DX-bone', 'DX-brain', 'DX-liver']]

# Define the RakEl algorithm
def rakel(X, y, k, m, strategy):
    n_labels = y.shape[1]
    ensemble = []
    labelsets = []

    if k > n_labels:
        raise ValueError("k must be smaller than or equal to the number of labels")

    if strategy not in ['disjoint', 'overlapping']:
        raise ValueError("strategy must be either 'disjoint' or 'overlapping'")

    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 4
    }

    for i in range(m):
        if strategy == 'disjoint':
            chunks = [list(range(n_labels))[i:i + k] for i in range(0, n_labels, k)]
            labelset_indices = random.choice(chunks)
        else:
            labelset_indices = random.sample(list(range(n_labels)), k)

        labelset = y.columns[labelset_indices]
        labelsets.append(labelset)

        clf = RandomForestClassifier(**params)
        clf.fit(X, y[labelset])
        ensemble.append(clf)

    return ensemble, labelsets

# Function to predict using RakEl
def predict_rakel(X, ensemble, labelsets, y_columns):
    n_labels = len(y_columns)
    y_pred = np.zeros((X.shape[0], n_labels))

    for clf, labelset in zip(ensemble, labelsets):
        labelset_indices = y_columns.get_indexer(labelset)  # Use the passed y_columns for indexing
        y_pred[:, labelset_indices] = clf.predict(X)

    return y_pred

# Train RakEl on LungCancer32.csv training data
ensemble, labelsets = rakel(X_train, y_train, k=3, m=10, strategy='disjoint')







y_pred = predict_rakel(X_test, ensemble, labelsets, y_test.columns)









# Predict on the LungCancer06.csv (verification dataset)
y_verification_pred = predict_rakel(X_verification, ensemble, labelsets, y_train.columns)

# Evaluate the performance on the verification data using multilabel metrics
verification_hamming = hamming_loss(y_verification, y_verification_pred)
verification_f1_micro = f1_score(y_verification, y_verification_pred, average='micro')
verification_f1_macro = f1_score(y_verification, y_verification_pred, average='macro')
verification_precision_micro = precision_score(y_verification, y_verification_pred, average='micro')
verification_precision_macro = precision_score(y_verification, y_verification_pred, average='macro')
verification_recall_micro = recall_score(y_verification, y_verification_pred, average='micro')
verification_recall_macro = recall_score(y_verification, y_verification_pred, average='macro')
acc_verification = accuracy_score(y_verification, y_verification_pred)
print(f"Average Accuracy (Verification Set): {acc_verification}")
print(f"Hamming Loss on verification data (LungCancer06.csv): {verification_hamming:.4f}")
print(f"Micro F1-Score on verification data: {verification_f1_micro:.4f}")
print(f"Macro F1-Score on verification data: {verification_f1_macro:.4f}")
print(f"Micro Precision on verification data: {verification_precision_micro:.4f}")
print(f"Macro Precision on verification data: {verification_precision_macro:.4f}")
print(f"Micro Recall on verification data: {verification_recall_micro:.4f}")
print(f"Macro Recall on verification data: {verification_recall_macro:.4f}")

# Count correct and wrong predictions for each label
labels = ['DX-bone', 'DX-brain', 'DX-liver']

for i, label in enumerate(labels):
    correct_count = (y_verification.iloc[:, i] == y_verification_pred[:, i]).sum()
    wrong_count = (y_verification.iloc[:, i] != y_verification_pred[:, i]).sum()
    print(f"Label: {label}")
    print(f"Correct predictions: {correct_count}")
    print(f"Wrong predictions: {wrong_count}")
    print("-" * 30)

# Visualize correct and wrong predictions for verification data (overall)
correct_predictions = (y_verification == y_verification_pred).all(axis=1)
wrong_predictions = ~correct_predictions

# Plot
plt.figure(figsize=(8, 6))
plt.bar(['Correct', 'Wrong'], [correct_predictions.sum(), wrong_predictions.sum()])
plt.title("Correct vs Wrong Predictions on Verification Data")
plt.ylabel("Number of Predictions")
plt.show()


print("########################################################")

# Precision-Recall Curve for each label on the verification set
from sklearn.metrics import precision_recall_curve

for i in range(y_verification.shape[1]):
    precision, recall, _ = precision_recall_curve(y_verification.iloc[:, i], y_verification_pred[:, i])
    plt.plot(recall, precision, label=f"Label {labels[i]}")  # Use actual label names
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve (Verification Set)")
plt.show()

#*******************************************************************************
# Count correct and wrong predictions for each label
labels = ['DX-bone', 'DX-brain', 'DX-liver']

# Convert y_verification_pred to a dense array (if it is not already)
y_verification_pred_dense = y_verification_pred

for i, label in enumerate(labels):
    correct_count = (y_verification.iloc[:, i].values == y_verification_pred_dense[:, i]).sum()
    wrong_count = (y_verification.iloc[:, i].values != y_verification_pred_dense[:, i]).sum()
    total_count = correct_count + wrong_count
    accuracy = (correct_count / total_count) * 100

    print(f"Label: {label}")
    print(f"Correct predictions: {correct_count}")
    print(f"Wrong predictions: {wrong_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 30)

# Visualize correct and wrong predictions for verification data (overall)
correct_predictions = (y_verification.values == y_verification_pred_dense).all(axis=1)
wrong_predictions = ~correct_predictions

# Plot
plt.figure(figsize=(8, 6))
plt.bar(['Correct', 'Wrong'], [correct_predictions.sum(), wrong_predictions.sum()])
plt.title("Correct vs Wrong Predictions on Verification Data (RakEL_RF )")
plt.ylabel("Number of Predictions")
plt.show()

# ##################################################
print("The Test Set:")
print("")

# Import necessary libraries
import numpy as np

# Function to compute correct and wrong counts
def compute_correct_wrong(y_true, y_pred_dense, labels):
    correct_counts = []
    wrong_counts = []
    
    for i, label in enumerate(labels):
        correct_count = (y_true.iloc[:, i].values == y_pred_dense[:, i]).sum()
        wrong_count = (y_true.iloc[:, i].values != y_pred_dense[:, i]).sum()
        correct_counts.append(correct_count)
        wrong_counts.append(wrong_count)
    
    return correct_counts, wrong_counts

# Labels for the tasks
labels = ['DX-bone', 'DX-brain', 'DX-liver']

# Convert y_pred and y_verification_pred to dense arrays (if they are not already)
y_pred_dense = y_pred
y_verification_pred_dense = y_verification_pred

# Compute correct and wrong predictions for both test and verification sets
correct_counts_test, wrong_counts_test = compute_correct_wrong(y_test, y_pred_dense, labels)
correct_counts_verification, wrong_counts_verification = compute_correct_wrong(y_verification, y_verification_pred_dense, labels)


print("Correct Test:", correct_counts_test)
print("Wrong Test:", wrong_counts_test)
print("Correct Verification:", correct_counts_verification)
print("Wrong Verification:", wrong_counts_verification)
 
# Create a diverging bar chart (similar to a population pyramid)
fig, ax = plt.subplots(figsize=(8, 6))

# Define bar width and position
bar_width = 0.3
positions = np.arange(len(labels))

# Plot for test set
ax.barh(positions, -np.array(correct_counts_test), color='blue', height=bar_width, label='Correct (Test Set)')
ax.barh(positions + bar_width, wrong_counts_test, color='green', height=bar_width, label='Wrong (Test Set)')

# Plot for verification set
ax.barh(positions + bar_width * 2, -np.array(correct_counts_verification), color='cyan', height=bar_width, label='Correct (Verification Set)')
ax.barh(positions + bar_width * 3, wrong_counts_verification, color='orange', height=bar_width, label='Wrong (Verification Set)')

# Set the labels and title
ax.set_yticks(positions + bar_width * 1.5)
ax.set_yticklabels(labels)
ax.set_xlabel("Number of Predictions")
ax.set_title("Correct vs Wrong Predictions for Test and Verification Sets (RakEL_RF)")

# Add legend
ax.legend()

# Show plot
plt.show()

# Plot
plt.figure(figsize=(8, 6))
plt.bar(['Correct', 'Wrong'], [correct_predictions.sum(), wrong_predictions.sum()])
plt.title("Correct vs Wrong Predictions on Verification Data (RakEL_RF)")
plt.ylabel("Number of Predictions")
plt.show()




print("##########################################################")

# Calculate and print correct and wrong predictions for the test set
y_pred_dense = y_pred  # Convert test set predictions to a dense array

for i, label in enumerate(labels):
    correct_count = (y_test.iloc[:, i].values == y_pred_dense[:, i]).sum()
    wrong_count = (y_test.iloc[:, i].values != y_pred_dense[:, i]).sum()
    total_count = correct_count + wrong_count
    accuracy = (correct_count / total_count) * 100

    print(f"Label: {label} (Test Set)")
    print(f"Correct predictions: {correct_count}")
    print(f"Wrong predictions: {wrong_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 30)

# Calculate and print correct and wrong predictions for the verification set
for i, label in enumerate(labels):
    correct_count = (y_verification.iloc[:, i].values == y_verification_pred_dense[:, i]).sum()
    wrong_count = (y_verification.iloc[:, i].values != y_verification_pred_dense[:, i]).sum()
    total_count = correct_count + wrong_count
    accuracy = (correct_count / total_count) * 100

    print(f"Label: {label} (Verification Set)")
    print(f"Correct predictions: {correct_count}")
    print(f"Wrong predictions: {wrong_count}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-" * 30)
    



# Count correct and wrong predictions for the verification set
correct_count_verification = []
wrong_count_verification = []

for i, label in enumerate(labels):
    correct = (y_verification.iloc[:, i].values == y_verification_pred_dense[:, i]).sum()
    wrong = (y_verification.iloc[:, i].values != y_verification_pred_dense[:, i]).sum()
    correct_count_verification.append(correct)
    wrong_count_verification.append(wrong)

# Create a diverging bar chart (like a population pyramid) for the verification set
fig, ax = plt.subplots(figsize=(8, 6))

# Define bar width and position
bar_width = 0.3
positions = np.arange(len(labels))

# Plot for verification set
ax.barh(positions, -np.array(correct_count_verification), color='cyan', height=bar_width, label='Correct (Verification Set)')
ax.barh(positions + bar_width, wrong_count_verification, color='orange', height=bar_width, label='Wrong (Verification Set)')

# Set the labels and title
ax.set_yticks(positions + bar_width / 2)
ax.set_yticklabels(labels)
ax.set_xlabel("Number of Predictions")
ax.set_title("Correct vs Wrong Predictions for Verification Set (RakEL_RF)")

# Add legend
ax.legend()

# Show plot
plt.show()





print("##################################################")
print("The Test Set:")
print("")

# Count correct and wrong predictions for the test set
correct_count_test = []
wrong_count_test = []

for i, label in enumerate(labels):
    correct = (y_test.iloc[:, i].values == y_pred_dense[:, i]).sum()
    wrong = (y_test.iloc[:, i].values != y_pred_dense[:, i]).sum()
    correct_count_test.append(correct)
    wrong_count_test.append(wrong)

# Create a diverging bar chart (like a population pyramid) for the test set
fig, ax = plt.subplots(figsize=(8, 6))

# Define bar width and position
bar_width = 0.3
positions = np.arange(len(labels))

# Plot for test set
ax.barh(positions, -np.array(correct_count_test), color='blue', height=bar_width, label='Correct (Test Set)')
ax.barh(positions + bar_width, wrong_count_test, color='green', height=bar_width, label='Wrong (Test Set)')

# Set the labels and title
ax.set_yticks(positions + bar_width / 2)
ax.set_yticklabels(labels)
ax.set_xlabel("Number of Predictions")
ax.set_title("Correct vs Wrong Predictions for Test Set (RakEL_RF)")

# Add legend
ax.legend()

# Show plot
plt.show()

colors = ['blue', 'cyan', 'green', 'orange']

# Compute correct and wrong predictions
def compute_correct_wrong(y_true, y_pred_dense):
    correct_counts = (y_true.values == y_pred_dense).sum(axis=1)
    wrong_counts = (y_true.values != y_pred_dense).sum(axis=1)
    return correct_counts, wrong_counts

# Get correct and wrong counts for the test and verification sets
correct_counts_test, wrong_counts_test = compute_correct_wrong(y_test, y_pred_dense)
correct_counts_verification, wrong_counts_verification = compute_correct_wrong(y_verification, y_verification_pred_dense)

# Print correct and wrong predictions for each label
print("Correct and Wrong Predictions for Test Set:")
for i, label in enumerate(labels):
    correct_test = (y_test.iloc[:, i].values == y_pred_dense[:, i]).sum()
    wrong_test = (y_test.iloc[:, i].values != y_pred_dense[:, i]).sum()
    
    print(f"Label: {label}")
    print(f"Correct (Test Set): {correct_test}")
    print(f"Wrong (Test Set): {wrong_test}")
    print("-" * 30)

print("\nCorrect and Wrong Predictions for Verification Set:")
for i, label in enumerate(labels):
    correct_verification = (y_verification.iloc[:, i].values == y_verification_pred_dense[:, i]).sum()
    wrong_verification = (y_verification.iloc[:, i].values != y_verification_pred_dense[:, i]).sum()
    
    print(f"Label: {label}")
    print(f"Correct (Verification Set): {correct_verification}")
    print(f"Wrong (Verification Set): {wrong_verification}")
    print("-" * 30)

# Plot overall correct and wrong predictions for both sets
plt.figure(figsize=(10, 6))

# Plot for test set
plt.bar(['Correct (Test)',  'Correct (Verification)', 'Wrong (Test)', 'Wrong (Verification)'],
        [correct_counts_test.sum(), correct_counts_verification.sum(),
         wrong_counts_test.sum(), wrong_counts_verification.sum()], color=colors)

# Set labels and title
plt.title("Correct vs Wrong Predictions for Test and Verification Sets (RakEL_RF)")
plt.ylabel("Number of Predictions")
plt.show()


print ("Correct T: ", correct_counts_test.sum())
print ("Wrong T: ", wrong_counts_test.sum())
print ("Correct V: ", correct_counts_verification.sum())
print ("Wrong V: ", wrong_counts_verification.sum())










import numpy as np

# Variance of predictions for the test set
y_pred_variance = np.var(y_pred, axis=0)
print("Variance of Predictions (Test Set) per label:")
for i, var in enumerate(y_pred_variance):
    print(f"Label {i + 1}: {var:.4f}")

# Variance of predictions for the verification set
y_verification_variance = np.var(y_verification_pred, axis=0)
print("Variance of Predictions (Verification Set) per label:")
for i, var in enumerate(y_verification_variance):
    print(f"Label {i + 1}: {var:.4f}")

