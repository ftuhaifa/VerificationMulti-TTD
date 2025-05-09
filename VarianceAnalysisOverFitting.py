# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:53:04 2024

@author: ftuha
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
#main_data = pd.read_csv("LungCancer25.csv")
#verified_data = pd.read_csv("encoded_merged_data03.csv")

main_data = pd.read_csv('LungCancer32.csv')
verified_data = pd.read_csv('LungCancer06.csv')

# Select relevant features; ensure both datasets contain these features
features = ['Age', 'Sex', 'Primary_Site', 'Laterality', 'Histology']
main_data = main_data[features]
verified_data = verified_data[features]

# You can also handle missing values or apply transformations if necessary
# For example, fill missing values if any
main_data.fillna(main_data.mean(), inplace=True)
verified_data.fillna(verified_data.mean(), inplace=True)

# Calculate variance for each feature in both datasets
train_variance = main_data.var()
verify_variance = verified_data.var()

# Print the variance values
print("Training Data Variance:\n", train_variance)
print("Verification Data Variance:\n", verify_variance)

# Create a DataFrame for easier plotting
variance_df = pd.DataFrame({'Training Variance': train_variance,
                            'Verification Variance': verify_variance})

# Plotting the variance data
variance_df.plot(kind='bar', figsize=(10, 6))
plt.title('Comparison of Variance between Training and Verification Data')
plt.ylabel('Variance')
plt.xlabel('Features')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
