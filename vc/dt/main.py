import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import display_data as dd
from sklearn.base import BaseEstimator, ClassifierMixin

plt.rcParams.update({'font.size': 14})

"""
    Class that extends BaseEstimator and ClassifierMixin
    in order to be available for using with voting classifier.

    Implements decision tree algorithm trained using
    grid-search and cross-validation.
"""


class DecisionTreeVC(BaseEstimator, ClassifierMixin):
    def __init__(self, save_folder='.'):
        self.classifier = None
        self.save_folder = save_folder
        self.classes_ = None

    """
        Function to fit the model using provided training data
        and grid-search space.
        Grid-search with cross-validation.
    """

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        dt_classifier = DecisionTreeClassifier(random_state=42)

        param_grid = {
            'max_leaf_nodes': range(10, 21, 5),
            'max_features': range(2, 9, 3),
            'max_depth': range(6, 11, 2),
            'min_samples_split': range(10, 21, 10),
            'min_samples_leaf': range(5, 11, 5)
        }

        # Uses CV to evaluate each parameter combination
        print('Decision tree GS running...')
        grid_search = GridSearchCV(estimator=dt_classifier,
                                   param_grid=param_grid, cv=5, verbose=1)
        grid_search.fit(X, y)
        best_dt_classifier = grid_search.best_estimator_

        print("Best Estimator's Hyperparameters:",
              best_dt_classifier.get_params())

        with open(self.save_folder + f'/grid_search_best.txt', 'w') as self.file:
            for key, value in best_dt_classifier.get_params().items():
                self.file.write(f'{key}: {value}\n')

        k = 5
        cv_scores = cross_val_score(
            best_dt_classifier, X, y, cv=k, scoring='accuracy')
        print(
            f'Cross validation with best grid search hyperparameters: {np.mean(cv_scores)}')
        sc = {'test_score': cv_scores}
        dd.visualize_cv(k, sc, self.save_folder)

        self.classifier = best_dt_classifier

    """
        Tests model on provided data.
    """

    def test(self, X, y):
        if self.classifier is None:
            print(f'Classifier needs to be fitted first!')
            return

        y_test_pred = self.classifier.predict(X)
        dd.visualize_cr_cm(y, y_test_pred, self.folder)

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

    """
        Returns model prediction for provided instance.
    """

    def predict(self, X):
        if self.classifier is None:
            print(f'Classifier needs to be fitted first!')
            return

        return self.classifier.predict(X)
    
    """
        Returns model prediction probability for provided instance.
    """

    def predict_proba(self, X):
        if self.classifier is None:
            print(f'Classifier needs to be fitted first!')
            return

        return self.classifier.predict_proba(X)
