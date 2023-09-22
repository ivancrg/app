import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../new_encoded_norm0367.csv')

train_data, remaining_data = train_test_split(
    data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(
    remaining_data, test_size=0.35, random_state=42)

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_valid, y_valid = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

# clf = MLPClassifier(random_state=22, max_iter=300, early_stopping=True)
clf = MLPClassifier(random_state=42, early_stopping=True, hidden_layer_sizes=(512, 512, 256, 256, 128))
clf.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))

print(clf.score(X_test, y_test))
