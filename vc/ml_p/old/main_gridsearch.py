import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score

data = pd.read_csv('../new_encoded_norm0367.csv')

train_data, remaining_data = train_test_split(
    data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(
    remaining_data, test_size=0.35, random_state=42)

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_valid, y_valid = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]

X_train_valid = pd.concat([X_train, X_valid])
y_train_valid = pd.concat([y_train, y_valid])

# Define maximum number of hidden layers
max_layers = 5
# Define available neuron sizes
neuron_sizes = [1] + list(range(5, 106, 20))

best_accuracy = 0.0
best_params = None

# Generate all combinations of hidden layers' neuron sizes
combinations = []
for num_layers in range(1, max_layers + 1):
    layer_combinations = itertools.product(neuron_sizes, repeat=num_layers)
    combinations.extend(layer_combinations)

for c in combinations:
    clf = MLPClassifier(random_state=42, hidden_layer_sizes=c, early_stopping=True)
    
    # Perform cross-validation
    scores = cross_val_score(clf, X_train_valid, y_train_valid, cv=5)
    average_accuracy = np.mean(scores)
    
    # Check if the current combination is the best
    if average_accuracy > best_accuracy:
        best_accuracy = average_accuracy
        best_params = c
    
    print(f"Layers: {c}, Accuracy: {average_accuracy}")

print("Best parameters:", best_params)
print("Best accuracy:", best_accuracy)

# Best parameters: (3, 95) relu
# Best accuracy: 0.8152106885919836

# Best parameters: (3, 95) tanh
# Best accuracy: 0.8152106885919836