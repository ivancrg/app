import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
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

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

param_grid = {
    'n_estimators': range(25, 101, 25),
    'max_leaf_nodes': range(10, 41, 10),
    'max_features': range(2, 12, 3),
    'max_depth': range(6, 11, 2),
    'min_samples_split': range(1, 22, 10),
    'min_samples_leaf': range(1, 12, 5)
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train_valid, y_train_valid)
best_rf_classifier = grid_search.best_estimator_

print("Best Estimator's Hyperparameters:")
print("max_leaf_nodes:", best_rf_classifier.get_params()['max_leaf_nodes'])
print("max_features:", best_rf_classifier.get_params()['max_features'])
print("max_depth:", best_rf_classifier.get_params()['max_depth'])
print("min_samples_split:", best_rf_classifier.get_params()['min_samples_split'])
print("min_samples_leaf:", best_rf_classifier.get_params()['min_samples_leaf'])

with open(folder + f'/rf_gs_best_params.txt', 'w') as file:
    for key, value in best_rf_classifier.get_params().items():
        file.write(f'{key}: {value}\n')

k = 5
cv_scores = cross_val_score(best_rf_classifier, X_train_valid, y_train_valid, cv=k, scoring='accuracy')
print(f'Cross validation with best grid search hyperparameters: {np.mean(cv_scores)}')
sc = {'test_score': cv_scores}
dd.visualize_cv(k, sc, folder, 'rf_gs_')


rf_classifier = best_rf_classifier

y_test_pred = rf_classifier.predict(X_test)
dd.visualize_cr_cm(y_test, y_test_pred, folder, 'rf_gs_')

plt.show()

plt.figure(figsize=(12, 10))

results = permutation_importance(
    rf_classifier,
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
plt.savefig(folder + '/rf_gs_fimp.png')