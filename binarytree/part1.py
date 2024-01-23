import pandas as pd
import numpy as np
from math import log2

# Step 1: Handle continuous-valued attributes
def handle_continuous_attributes(dataset):
    for col in dataset.columns:
        if dataset[col].dtype == 'float64':
            dataset[col].fillna(dataset[col].mean(), inplace=True)
    return dataset

# Step 2: Check for errors in the dataset
def check_dataset_errors(dataset):
    if dataset.isnull().values.any():
        print("Dataset contains missing values.")
    if dataset.duplicated().any():
        print("Dataset contains redundant or repeated input samples.")
    output_column = dataset.columns[-1]
    unique_pairs = dataset.groupby(dataset.columns[:-1].tolist())[output_column].nunique()
    contradicting_pairs = unique_pairs[unique_pairs > 1]
    if not contradicting_pairs.empty:
        print("Dataset contains contradicting <input, output> pairs.")

# Step 3: Implement ID3 algorithm
def calculate_entropy(data):
    target_col = data.columns[-1]
    entropy = 0
    values = data[target_col].unique()
    for value in values:
        p = data[target_col].value_counts()[value] / len(data)
        entropy += -p * log2(p)
    return entropy

def calculate_information_gain(data, attribute):
    target_col = data.columns[-1]
    total_entropy = calculate_entropy(data)
    values = data[attribute].unique()
    weighted_entropy = 0

    for value in values:
        subset = data[data[attribute] == value]
        subset_entropy = calculate_entropy(subset)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain

# Main Function
def my_ID3(dataset, parent_node_class=None):
    if len(dataset.columns) == 1:
        return parent_node_class
    elif len(dataset[dataset.columns[-1]].unique()) == 1:
        return dataset[dataset.columns[-1]].unique()[0]
    
    attributes = dataset.columns[:-1]
    gains = [calculate_information_gain(dataset, attr) for attr in attributes]
    selected_attribute = attributes[np.argmax(gains)]

    print(f"Attribute {selected_attribute} with Gain = {max(gains)} is chosen as the decision attribute.")

    tree = {selected_attribute: {}}
    for value in dataset[selected_attribute].unique():
        subset = dataset[dataset[selected_attribute] == value]

        if len(subset) == 0:
            tree[selected_attribute][value] = parent_node_class
        elif len(subset[subset.columns[-1]].unique()) == 1:
            tree[selected_attribute][value] = subset[subset.columns[-1]].unique()[0]
        else:
            tree[selected_attribute][value] = my_ID3(subset, subset[subset.columns[-1]].mode()[0])

    return tree

# Load the dataset from CSV
dataset = pd.read_csv("lab01_dataset_1.csv")

# Step 1: Handle continuous-valued attributes
dataset = handle_continuous_attributes(dataset)
print("Updated dataset after handling continuous-valued attributes:")
print(dataset)

# Step 2: Check for errors in the dataset
check_dataset_errors(dataset)

# Step 3: Implement and run the ID3 algorithm
decision_tree = my_ID3(dataset)
print("Decision Tree:")
print(decision_tree)
