import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate

data = pd.read_csv('./final/postop_spl_cplx_othr_norm.csv')

train_data, remaining_data = train_test_split(
    data, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(
    remaining_data, test_size=0.35, random_state=42)

X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_valid, y_valid = valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

X_train_valid = pd.concat([X_train, X_valid])
y_train_valid = pd.concat([y_train, y_valid])

clf = MLPClassifier(random_state=42, hidden_layer_sizes=(
    512, 512, 256, 256, 128, 64), early_stopping=True, verbose=True)

k = 5
scores = cross_validate(clf, X_train_valid.to_numpy(), y_train_valid.to_numpy(),
                        cv=k, return_estimator=True)

for fold, score in enumerate(scores['test_score']):
    print(f"Fold {fold+1} accuracy: {score}")
print("Average accuracy:", np.mean(scores['test_score']))

plt.figure(1, figsize=(10, 6))
bars = plt.bar(range(1, k + 1), scores['test_score'])
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval,
             round(yval, 5), ha='center', va='bottom')

plt.xlabel('Fold')
plt.ylabel('Accuracy')

plt.show()
