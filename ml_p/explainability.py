import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from mlp_keras import MLP
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from PyALE import ale
import shap

NN_MODEL_NAME = '/nn_save_512_[512, 256, 128]_0.24_SGD_categorical_crossentropy_0.1'
SAVE = True

IS_MODEL_NN = True
OVERSAMPLED_VERSION = False
PREDICTION_INTERES = '/histology'

BASE_FOLDER = './report/OS_NN' if OVERSAMPLED_VERSION else './report/NO_OS'
FOLDER = BASE_FOLDER + PREDICTION_INTERES
FILE = '/data.csv'
FILE_NORM = '/data_norm.csv'
RF_BEST_MODEL_FILE = '/rf_gs_best_params.txt'

def parse_rf_params():
    params = {}
    with open(FOLDER + RF_BEST_MODEL_FILE, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            if key == 'class_weight' or key == 'max_samples' or key == 'n_jobs':
                continue
            # Convert values to appropriate data types (e.g., bool, int)
            if value == 'True':
                value = True
            elif value == 'False':
                value = False
            elif value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass  # Leave it as a string if not bool, int, or float
            params[key] = value

    print(params)
    return params

def finalize_method(exp_method, show=False, postfix='', fig=None):
    if show:
        plt.show()
    if SAVE:
        save(exp_method=exp_method, postfix=postfix, fig=fig)


def save(exp_method, postfix, fig=None):
    if IS_MODEL_NN:
        print("Saving " + FOLDER + f'/nn_expl_{exp_method}_{NN_MODEL_NAME[1:]}{postfix}.png')
        if fig is None:
            plt.savefig(
                FOLDER + f'/nn_expl_{exp_method}_{NN_MODEL_NAME[1:]}{postfix}.png')
        else:
            fig.savefig(
                FOLDER + f'/nn_expl_{exp_method}_{NN_MODEL_NAME[1:]}{postfix}.png')
    else:
        print("Saving " + FOLDER + f'/rf_gs_expl_{exp_method}{postfix}.png')
        if fig is None:
            plt.savefig(FOLDER + f'/rf_gs_expl_{exp_method}{postfix}.png')
        else:
            fig.savefig(FOLDER + f'/rf_gs_expl_{exp_method}{postfix}.png')

# Load data
data = pd.read_csv(FOLDER + FILE)
data_norm = pd.read_csv(FOLDER + FILE_NORM)
feature_names, target_names = data.columns[:-
                                           1].to_list(), data.columns[-1:].to_list()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

train_data_norm, test_data_norm = train_test_split(
    data_norm, test_size=0.2, random_state=42)
X_train_norm, y_train_norm = train_data_norm.iloc[:,
                                                  :-1], train_data_norm.iloc[:, -1]
X_test_norm, y_test_norm = test_data_norm.iloc[:,
                                               :-1], test_data_norm.iloc[:, -1]

# Load and train the model

model = None
if IS_MODEL_NN is False:
    # RF
    best_params = parse_rf_params()
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
else:
    # NN
    model = MLP(FOLDER + FILE_NORM)
    model.trained_model = tf.keras.models.load_model(FOLDER + NN_MODEL_NAME)


# PDP, ICE - centered
for c in y_train.unique():
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(14, 10))
    plt.title(f'Class {c}')
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.06, top=0.97)
    try:
        pdp = PartialDependenceDisplay.from_estimator(
            model,
            X_train_norm if IS_MODEL_NN else X_train,
            features=[i for i in range(len(X_train.columns))],
            target=c,
            method='brute',
            kind='both',
            centered=True,
            ax=ax
        )
        save('PDP_ICE', f'_{c}')
    except ValueError as ve:
        ax.text(
            0.5, 0.5, f'No instances classified as class {c}!', ha='center', va='center', fontsize=14)
        print("A ValueError occurred:", ve)
        save('PDP_ICE', f'_{c}')

plt.rcParams.update({'font.size': 14})

# ALE
for i, f in enumerate(X_train.columns.to_list()):
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.06, top=0.97)
    ale_eff = ale(
        X=X_train_norm if IS_MODEL_NN else X_train,
        model=model,
        feature=[f],
        grid_size=100,
        include_CI=True,
        fig=fig,
        ax=ax
    )
    finalize_method('ALE', f'_{f}', fig=fig, postfix=f'_{f.replace("/", "-")}')

# Feature importance
plt.figure(figsize=(12, 10))

def custom_scoring(est, X, y_true):
    y_pred = np.argmax(est.predict(X), axis=1)
    return accuracy_score(y_true.to_numpy(), y_pred)

results = permutation_importance(
    model.trained_model if IS_MODEL_NN else model,
    X_train_norm if IS_MODEL_NN else X_train,
    y_train_norm if IS_MODEL_NN else y_train,
    n_repeats=10,
    random_state=42,
    scoring=custom_scoring if IS_MODEL_NN else None)

importance = results.importances_mean

for i, v in enumerate(importance):
    print(f'Feature {i}: {v:.5f}')

plt.subplots_adjust(left=0.09, right=0.96, bottom=0.33, top=0.97)

plt.bar(X_test.columns, importance)
plt.xticks(rotation=90)
plt.ylabel('Importance')

finalize_method('f_imp')


# SHAP
# Clear the current figure (if any)
plt.clf()

# Close all open figures
plt.close('all')

if IS_MODEL_NN:
    shap_explainer = shap.KernelExplainer(
        model.trained_model, X_train_norm)
else:
    shap_explainer = shap.Explainer(model, X_train)


shap_values = shap_explainer.shap_values(
    X_test_norm.iloc[[0]] if IS_MODEL_NN else X_test.iloc[[0]], check_additivity=False)
shap.summary_plot(shap_values, X_test_norm.iloc[[
                  0]] if IS_MODEL_NN else X_test.iloc[[0]], show=False)
plt.subplots_adjust(left=0.33, right=0.86, bottom=0.12, top=0.97)
finalize_method('shap')
