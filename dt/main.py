import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import display_data as dd 

plt.rcParams.update({'font.size': 14})

folder = './report/ND/h2'
file = '/data.csv'

data = pd.read_csv(folder + file)

train_valid_data, test_data = train_test_split(
    data, test_size=0.2, random_state=42)

X_train_valid, y_train_valid = train_valid_data.iloc[:,
                                                     :-1], train_valid_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

dt_classifier = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_leaf_nodes': range(10, 21, 5),
    'max_features': range(2, 9, 3),
    'max_depth': range(6, 11, 2),
    'min_samples_split': range(10, 21, 10),
    'min_samples_leaf': range(5, 11, 5)
}

grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train_valid, y_train_valid)
best_dt_classifier = grid_search.best_estimator_

print("Best Estimator's Hyperparameters:")
print("max_leaf_nodes:", best_dt_classifier.get_params()['max_leaf_nodes'])
print("max_features:", best_dt_classifier.get_params()['max_features'])
print("max_depth:", best_dt_classifier.get_params()['max_depth'])
print("min_samples_split:", best_dt_classifier.get_params()['min_samples_split'])
print("min_samples_leaf:", best_dt_classifier.get_params()['min_samples_leaf'])

with open(folder + f'/dt_best_params.txt', 'w') as file:
    for key, value in best_dt_classifier.get_params().items():
        file.write(f'{key}: {value}\n')

k = 5
cv_scores = cross_val_score(best_dt_classifier, X_train_valid, y_train_valid, cv=k, scoring='accuracy')
print(f'Cross validation with best grid search hyperparameters: {np.mean(cv_scores)}')
sc = {'test_score': cv_scores}
dd.visualize_cv(k, sc, folder, 'dt_gs_')

dt_classifier = best_dt_classifier

y_test_pred = dt_classifier.predict(X_test)
dd.visualize_cr_cm(y_test, y_test_pred, folder, prefix='dt_gs_')

plt.show()

plt.figure(figsize=(12, 10))

results = permutation_importance(
    dt_classifier,
    X_train_valid,
    y_train_valid,
    n_repeats=10,
    random_state=42
    )

importance = results.importances_mean

for i, v in enumerate(importance):
    print(f'Feature {i}: {v:.5f}')

plt.subplots_adjust(left=0.09, right=0.96, bottom=0.33, top=0.97)

plt.bar(X_test.columns, importance)
plt.xticks(rotation=90)
plt.ylabel('Importance')
plt.savefig(folder + '/dt_gs_fimp.png')