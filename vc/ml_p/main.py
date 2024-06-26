import numpy as np
from vc.ml_p.mlp import MLP
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE

"""
    Class that extends BaseEstimator and ClassifierMixin
    in order to be available for using with voting classifier.

    Implements artificial neural network algorithm trained
    using grid-search and cross-validation.
"""


class MLPVC(BaseEstimator, ClassifierMixin):
    def __init__(self, save_folder=None, n_splits=5, binary=True, smote=False, verbose=False):
        self.save_folder = save_folder
        self.n_splits = n_splits
        self.binary = binary
        self.classifier = None
        self.smote = smote
        self.verbose = verbose

    """
        Function to fit the model using provided training data
        in the defined grid-search space.
        Grid-search with cross-validation.
    """

    def fit(self, X, y):
        losses = ['categorical_crossentropy', 'binary_crossentropy'] if self.binary else [
            'sparse_categorical_crossentropy', 'categorical_crossentropy']
        optimizers = ['Adam',
                        'SGD',
                      # 'AdamW',
                      # 'Adafactor',
                      # 'Nadam'
                      ]
        learning_rates = [0.0001]
        dropouts = [0.1]
        # learning_rates = [10 ** (-i) for i in range(1, 6)]
        # dropouts = np.linspace(0.05, 0.2, 4)

        print(self.save_folder)
        print(losses)
        print(optimizers)
        print(learning_rates)
        print(dropouts)

        mlp = MLP()
        log = pd.DataFrame()

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True)
        rlr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.0000001)

        skf = StratifiedKFold(n_splits=self.n_splits,
                              shuffle=True, random_state=42)
        splits = skf.split(X, y)
        folds = []

        for s in splits:
            ti, vi = s
            X_train_fold, X_val_fold = X[ti], X[vi]
            y_train_fold, y_val_fold = y[ti], y[vi]

            if self.smote:
                X_train_fold, y_train_fold = SMOTE().fit_resample(X_train_fold, y_train_fold)

            folds.append({
                'X_train_fold': X_train_fold,
                'y_train_fold': y_train_fold,
                'X_val_fold': X_val_fold,
                'y_val_fold': y_val_fold,
            })

        best = (0, 0, '')
        i = 0
        for loss in losses:
            for optimizer in optimizers:
                for learning_rate in learning_rates:
                    for dropout in dropouts:
                        avg_val_acc, avg_val_loss, train_accs, train_losses, val_accs, val_losses = 0, 0, [], [], [], []

                        mlp.set_params({
                            'optimizer_text': optimizer,
                            'learning_rate': learning_rate,
                            'callbacks': [early_stopping_callback, rlr_callback],
                            'dropout': dropout,
                            'loss': loss,
                        })

                        for fold in folds:
                            ta, tl, va, vl = mlp.fit_cv(
                                fold['X_train_fold'], fold['y_train_fold'], fold['X_val_fold'], fold['y_val_fold'], verbose=self.verbose)
                            train_accs.append(ta)
                            train_losses.append(tl)
                            val_accs.append(va)
                            val_losses.append(vl)

                        avg_val_acc, avg_val_loss = np.mean(
                            val_accs), np.mean(val_losses)

                        if avg_val_acc > best[0]:
                            best = avg_val_acc, avg_val_loss, mlp.get_name()

                        log = pd.concat([log,
                                        pd.DataFrame({
                                            'train_accuracy': ', '.join([str(ta) for ta in train_accs]),
                                            'train_loss': ', '.join([str(tl) for tl in train_losses]),
                                            'val_accuracy': ', '.join([str(va) for va in val_accs]),
                                            'val_loss': ', '.join([str(vl) for vl in val_losses]),
                                            'average_val_accuracy': [avg_val_acc],
                                            'average_val_loss': [avg_val_loss],
                                            'params': [mlp.get_params()]
                                        })],
                                        axis=0
                                        )

                        i += 1
                        print(val_accs, mlp.get_name())
                        print(best)
                        print(
                            f'Progress: {i}/{len(losses) * len(optimizers) * len(learning_rates) * len(dropouts)} ({round(i / (len(losses) * len(optimizers) * len(learning_rates) * len(dropouts)) * 100, 2)}%)\n')

        log.to_csv(f'{self.save_folder}/grid_search_log.csv', index=False)

        log.sort_values(by=['average_val_accuracy',
                        'average_val_loss'], ascending=False)

        bp = log.iloc[0]['params']

        mlp.set_params({
            'optimizer_text': bp['optimizer'],
            'learning_rate': bp['learning_rate'],
            'input_layer': bp['input_layer'],
            'hidden_layers': bp['hidden_layers'],
            'dropout': bp['dropout'],
            'loss': bp['loss'],
        })

        for fold in folds:
            ta, tl, va, vl = mlp.fit_cv(
                fold['X_train_fold'], fold['y_train_fold'], fold['X_val_fold'], fold['y_val_fold'], f'{self.save_folder}/cv.png', verbose=True)

        mlp.fit(X, y)
        self.classifier = mlp
