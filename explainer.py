import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.inspection import PartialDependenceDisplay

from PyALE import ale

from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

import shap


class Explainer:
    """
        Takes in a model for which model-agnostic method
        visualizations will be generated. Currently, PDP,
        ICE, ALE, SHAP and feature importance methods
        are supported. Predict and predict_proba functions
        need to be available on the given method. Data
        that will be used to generate visualizations must
        also be provided. If an additional data instance
        is provided, local explanation method visualizations
        will also be generated.

        Inputs:
            - model: any object that implements required functions
            - X: input data instances
            - y: ground truth predictions for X
            - features: list of strings representing feature names
            - save_folder: folder to save the generated figures in
            - verbose: whether to output additional logs or not
    """

    def __init__(self, model, X, y, features, save_folder=None, verbose=False):
        self.model = model
        self.X = X
        self.y = y
        self.features = features
        self.save_folder = save_folder
        self.verbose = verbose

        self.ensure_folder_exists(self.save_folder)

    def generate_pdp_ice(self):
        """
            Generates PDP/ICE explainability method visualization.
        """

        for c in np.unique(self.y):
            for i, feature in enumerate(self.features):
                plt.rcParams.update({'font.size': 12})
                fig, ax = plt.subplots(figsize=(12, 10))

                PartialDependenceDisplay.from_estimator(
                    self.model,
                    self.X,
                    features=[i],
                    feature_names=self.features,
                    target=c,
                    method='brute',
                    kind='both',
                    centered=True,
                    ax=ax
                )

                ax.set_title(f'{feature} PDP/ICE - Class {c}')

                self.save_figure(f'{feature}_{c}', subfolder='PDP_ICE')

    def generate_ale(self):
        """
            Generates ALE explainability method visualization.
        """

        if not isinstance(self.X, pd.DataFrame):
            X = pd.DataFrame(self.X, columns=self.features)
        else:
            X = self.X

        for feature in self.features:
            plt.rcParams.update({'font.size': 12})
            fig, ax = plt.subplots(figsize=(12, 10))
            fig.subplots_adjust(left=0.09, right=0.96, bottom=0.06, top=0.97)

            ale(
                X=X,
                model=self.model,
                feature=[feature],
                grid_size=100,
                include_CI=True,
                fig=fig,
                ax=ax
            )

            self.save_figure(feature, subfolder='ALE')

    def generate_fi(self):
        """
            Generates feature importance explainability method visualization.
        """

        def custom_scoring(est, X, y_true):
            return accuracy_score(y_true, est.predict(X))

        results = permutation_importance(
            self.model,
            self.X,
            self.y,
            n_repeats=10,
            random_state=42,
            scoring=custom_scoring)

        importance = results.importances_mean

        for i, v in enumerate(importance):
            print(f'Feature {i}: {v:.5f}')

        plt.subplots_adjust(left=0.09, right=0.96, bottom=0.33, top=0.97)

        plt.bar(self.features, importance)
        plt.xticks(rotation=90)
        plt.ylabel('Importance')

        self.save_figure('feature_importance')

    def generate_shap(self, Xi):
        """
            Generates SHAP explainability method visualization on the provided instance.
        """

        if Xi is None:
            print('explainer.py::generate_shap No data instance(s) provided.')
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        shap.initjs()

        if not hasattr(self, 'shap_explainer'):
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict, self.X, feature_names=self.features)

        shap_values = self.shap_explainer.shap_values(Xi)
        self.print_verbose(f'Shap values for {Xi}: {shap_values}')

        shap.summary_plot(shap_values, Xi, feature_names=self.features, show=False)

        if self.save_folder is not None:
            self.save_figure('shap')

    def print_verbose(self, output):
        if self.verbose:
            print(output)

    def ensure_folder_exists(self, folder):
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError as e:
                print(f"Creation of the directory '{folder}' failed. {e}")

    def save_figure(self, name, subfolder=None):
        folder = f'{self.save_folder}/{subfolder}' if subfolder is not None else self.save_folder

        self.ensure_folder_exists(folder)

        plt.savefig(f'{folder}/{name}.png')
