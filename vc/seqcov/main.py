import pandas as pd
from seqcov.sequential_covering import SequentialCovering
from sklearn.model_selection import StratifiedKFold
import numpy as np
import display_data as dd
from sklearn.base import BaseEstimator, ClassifierMixin

"""
    Class that extends BaseEstimator and ClassifierMixin
    in order to be available for using with voting classifier.

    Implements sequential covering algorithm trained using
    grid-search and cross-validation.
"""

class SequentialCoveringVC(BaseEstimator, ClassifierMixin):
    def __init__(self, X_labels, y_labels, save_folder='.', multiclass=False, k=5):
        self.save_folder = save_folder
        self.multiclass = multiclass
        self.k = k
        self.X_labels = X_labels
        self.y_labels = y_labels
        self.prediction_label = None
        self.data = None
        self.classifier = None

    """
        Runs cross-validation for provided data and parameters.
        Returns cross-validation scores.
    """
    def cv(self, X, y, params):
        skf = StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
        val_accuracies = []

        for fold_idx, fold in enumerate(skf.split(X, y)):
            train_index, val_index = fold
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]

            train_data = pd.concat([X_train_fold, pd.DataFrame(
                y_train_fold)], axis=1, ignore_index=True).reset_index().iloc[:, 1:]
            train_data.columns = self.data.columns

            valid_data = pd.concat([X_val_fold, pd.DataFrame(
                y_val_fold)], axis=1, ignore_index=True).reset_index().iloc[:, 1:]
            valid_data.columns = self.data.columns

            sc = SequentialCovering(
                train_data,
                multiclass=self.multiclass,
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                max_leaf_nodes=params['max_leaf_nodes'],
                output_name=self.prediction_label)
            sc.fit()

            preds = sc.predict_tmp(valid_data)
            mean_acc = (preds[self.prediction_label]
                        == preds['Prediction']).mean()
            val_accuracies.append(mean_acc)

        kf_scores = {'test_score': np.array(val_accuracies)}

        return (np.mean(kf_scores['test_score']), kf_scores)

    """
        Function to fit the model using provided training data
        and grid-search space.
        Grid-search with cross-validation.
    """

    def fit(self, X, y):
        self.data = pd.concat([pd.DataFrame(X, columns=self.X_labels),
                               pd.DataFrame(y, columns=self.y_labels)], axis=1)

        self.prediction_label = self.data.columns[-1]

        best_mean_acc, best_kf_scores, best_params = 0, {}, {}
        param_grid = {
            'max_leaf_nodes': range(10, 31, 5),
            'max_features': range(2, 12, 3),
            'max_depth': range(3, 15, 3),
            'min_samples_split': range(2, 10, 3),
            'min_samples_leaf': range(1, 12, 5)
        }

        n_iter = np.prod([len(value) for value in param_grid.values()])
        iter = 0

        for max_leaf_nodes in param_grid['max_leaf_nodes']:
            for max_features in param_grid['max_features']:
                for max_depth in param_grid['max_depth']:
                    for min_samples_split in param_grid['min_samples_split']:
                        for min_samples_leaf in param_grid['min_samples_leaf']:
                            iter += 1
                            print(f'Sequential covering {iter}/{n_iter}')

                            cur_gs_params = {
                                'max_leaf_nodes': max_leaf_nodes,
                                'max_features': max_features,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf
                            }

                            mean_acc, kf_scores = self.cv(
                                X=self.data.iloc[:, :-1],
                                y=self.data.iloc[:, -1],
                                params=cur_gs_params
                            )

                            if mean_acc > best_mean_acc:
                                best_mean_acc, best_kf_scores, best_params = mean_acc, kf_scores, cur_gs_params

        sc = SequentialCovering(
                self.data.copy(),
                multiclass=self.multiclass,
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                max_features=best_params['max_features'],
                max_leaf_nodes=best_params['max_leaf_nodes'],
                output_name=self.prediction_label)
        sc.fit()
        self.classifier = sc

        dd.visualize_cv(self.k, best_kf_scores, self.save_folder)
        with open(self.save_folder + f'/grid_search_best.txt', 'w') as file:
            for key, value in best_params.items():
                file.write(f'{key}: {value}\n')

    """
        Returns model prediction for provided instance.
    """

    def predict(self, X):
        X = pd.DataFrame(X, columns=self.X_labels)

        if self.classifier is None:
            print(f'Classifier needs to be fitted first!')
            return

        return self.classifier.predict(X).flatten()