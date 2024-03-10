import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import display_data as dd
from sklearn.base import BaseEstimator, ClassifierMixin

plt.rcParams.update({'font.size': 14})


class RandomForestVC(BaseEstimator, ClassifierMixin):
    def __init__(self, save_folder='.'):
        self.classifier = None
        self.save_folder = save_folder

    """
        Function to fit the model using provided training data.
        Grid-search with cross-validation.
    """

    def fit(self, X, y):
        rf_classifier = RandomForestClassifier(random_state=42)

        param_grid = {
            'n_estimators': range(25, 101, 25),
            'max_leaf_nodes': range(10, 41, 10),
            'max_features': range(2, 12, 3),
            'max_depth': range(6, 11, 2),
            'min_samples_split': range(1, 22, 10),
            'min_samples_leaf': range(1, 12, 5)
        }

        # Uses CV to evaluate each parameter combination
        print("Random forest GS running...")
        grid_search = GridSearchCV(
            estimator=rf_classifier, param_grid=param_grid, cv=5, verbose=1)
        grid_search.fit(X, y)
        best_rf_classifier = grid_search.best_estimator_

        print("Best Estimator's Hyperparameters:",
              best_rf_classifier.get_params())

        with open(self.save_folder + f'/grid_search_best.txt', 'w') as self.file:
            for key, value in best_rf_classifier.get_params().items():
                self.file.write(f'{key}: {value}\n')

        k = 5
        cv_scores = cross_val_score(
            best_rf_classifier, X, y, cv=k, scoring='accuracy')
        print(
            f'Cross validation with best grid search hyperparameters: {np.mean(cv_scores)}')
        sc = {'test_score': cv_scores}
        dd.visualize_cv(k, sc, self.save_folder)

        self.classifier = best_rf_classifier

    def test(self, X, y):
        if self.classifier is None:
            print(f'Classifier needs to be fitted first!')
            return

        y_test_pred = self.classifier.predict(X)
        dd.visualize_cr_cm(y, y_test_pred, self.save_folder)

        plt.show()

        plt.figure(figsize=(12, 10))

        results = permutation_importance(
            self.classifier,
            X,
            y,
            n_repeats=10,
            random_state=42
        )

        importance = results.importances_mean

        for i, v in enumerate(importance):
            print(f'Feature {i}: {v:.5f}')

        plt.subplots_adjust(left=0.09, right=0.96, bottom=0.33, top=0.97)

        plt.bar(X.columns, importance)
        plt.xticks(rotation=90)
        plt.ylabel('Importance')
        plt.savefig(self.save_folder + '/feature_importance.png')

    def predict(self, X):
        if self.classifier is None:
            print(f'Classifier needs to be fitted first!')
            return

        return self.classifier.predict(X).flatten()
