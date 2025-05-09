# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 08:06:27 2024

@author: ftuha
"""

# Import the function
from sklearn.datasets import make_multilabel_classification
from skmultilearn.problem_transform import LabelPowerset
from sklearn.ensemble import RandomForestClassifier
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, hamming_loss, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

# Load data
data_train = pd.read_csv('LungCancer32.csv')

X = data_train[['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology']]
y = data_train[['DX-bone', 'DX-brain', 'DX-liver']]

# Normalize the data using Min-Max Normalization
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y,
                                                    test_size=0.3,
                                                    random_state=42)

# Load the verification dataset (LungCancer06.csv)
data_verification = pd.read_csv('LungCancer06.csv')

X_verificationS = data_verification[['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology']]
y_verification = data_verification[['DX-bone', 'DX-brain', 'DX-liver']]

X_verification = scaler.transform(X_verificationS)

# Initialize LabelPowerset multi-label classifier with Random Forest
classifier = LabelPowerset(classifier=RandomForestClassifier(),
                           require_dense=[False, True])

# Hyperparameter tuning (optional step)
param_grid = {'classifier__n_estimators': [200],
              'classifier__max_depth': [10],
              'classifier__min_samples_split': [2]}

# Train
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Import the metrics
from sklearn.metrics import hamming_loss, f1_score, precision_recall_curve, accuracy_score

# Calculate metrics for the test set
hl = hamming_loss(y_test, y_pred)
print(f"Hamming Loss (Test Set): {hl}")

f1_macro = f1_score(y_test, y_pred, average="macro")
print(f"Macro F1-Score (Test Set): {f1_macro}")

f1_micro = f1_score(y_test, y_pred, average="micro")
print(f"Micro F1-Score (Test Set): {f1_micro}")

acc = accuracy_score(y_test, y_pred)
print(f"Average Accuracy (Test Set): {acc}")

# Precision-Recall Curve for each label on the test set
for i in range(y.shape[1]):
    precision, recall, _ = precision_recall_curve(y_test.iloc[:, i], y_pred.toarray()[:, i])
    plt.plot(recall, precision, label=f"Label {i+1}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve (Test Set)")
plt.show()

# Predict on the verification set
y_verification_pred = classifier.predict(X_verification)

# Calculate metrics for the verification set
hl_verification = hamming_loss(y_verification, y_verification_pred)
print(f"Hamming Loss (Verification Set): {hl_verification}")

f1_macro_verification = f1_score(y_verification, y_verification_pred, average="macro")
print(f"Macro F1-Score (Verification Set): {f1_macro_verification}")

f1_micro_verification = f1_score(y_verification, y_verification_pred, average="micro")
print(f"Micro F1-Score (Verification Set): {f1_micro_verification}")

acc_verification = accuracy_score(y_verification, y_verification_pred)
print(f"Average Accuracy (Verification Set): {acc_verification}")

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

# Precision-Recall Curve for each label on the verification set
for i in range(y_verification.shape[1]):
    precision, recall, _ = precision_recall_curve(y_verification.iloc[:, i], y_verification_pred.toarray()[:, i])
    plt.plot(recall, precision, label=f"Label {i+1}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve (Verification Set)")
plt.show()



#*******************************************************************************
# Count correct and wrong predictions for each label
labels = ['DX-bone', 'DX-brain', 'DX-liver']

# Convert y_verification_pred to a dense array (if it is not already)
y_verification_pred_dense = y_verification_pred.toarray()

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
plt.title("Correct vs Wrong Predictions on Verification Data")
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
ax.set_title("Correct vs Wrong Predictions for Verification Set (LP-RF)")

# Add legend
ax.legend()

# Show plot
plt.show()




print("##################################################")
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
y_pred_dense = y_pred.toarray()
y_verification_pred_dense = y_verification_pred.toarray()

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
ax.set_title("Correct vs Wrong Predictions for Test and Verification Sets (LP-RF)")

# Add legend
ax.legend()

# Show plot
plt.show()


print("##########################################################")
print("##########################################################")

# Calculate and print correct and wrong predictions for the test set
y_pred_dense = y_pred.toarray()  # Convert test set predictions to a dense array

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
ax.set_title("Correct vs Wrong Predictions for Test Set (LP-RF)")

# Add legend
ax.legend()

# Show plot
plt.show()


# Variance of predictions for the test set
y_pred_variance = np.var(y_pred.toarray(), axis=0)
print("Variance of Predictions (Test Set) per label:")
for i, var in enumerate(y_pred_variance):
    print(f"Label {i + 1}: {var:.4f}")

# Variance of predictions for the verification set
y_verification_variance = np.var(y_verification_pred.toarray(), axis=0)
print("Variance of Predictions (Verification Set) per label:")
for i, var in enumerate(y_verification_variance):
    print(f"Label {i + 1}: {var:.4f}")



# Variance of Predictions for Test Data
y_pred_variance = np.var(y_pred.toarray(), axis=0)
print("Variance of Predictions (Test Set) per label:")
for i, var in enumerate(y_pred_variance):
    print(f"Label {i + 1}: {var:.4f}")

# Variance of Predictions for Verification Data
y_verification_variance = np.var(y_verification_pred.toarray(), axis=0)
print("Variance of Predictions (Verification Set) per label:")
for i, var in enumerate(y_verification_variance):
    print(f"Label {i + 1}: {var:.4f}")




from sklearn.model_selection import learning_curve

# Function to plot learning curves
def plot_learning_curve(estimator, X, y, X_verification, y_verification, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=5,
        scoring='f1_micro',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5),
        random_state=42
    )
    
    # Calculate mean and std for training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, test_mean, label='Cross-validation Score', marker='s')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

    # Performance on Verification Set
    verification_score = []
    for size in train_sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split(X, y, train_size=size, random_state=42)
        estimator.fit(X_train_subset, y_train_subset)
        y_ver_pred = estimator.predict(X_verification)
        verification_score.append(f1_score(y_verification, y_ver_pred, average="micro"))
    
    plt.plot(train_sizes, verification_score, label='Verification Set', marker='^', linestyle='--')

    # Plot settings
    plt.title(title)
    plt.xlabel('Training Examples')
    plt.ylabel('F1-Score')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Plot the learning curve
plot_learning_curve(classifier, X_train, y_train, X_verification, y_verification, 
                    title="Learning Curve for RandomForest + LabelPowerset")








