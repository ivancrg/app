import pandas as pd
import oner
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import display_data as dd

FOLDER = './report/NO_OS/histology_binary'
FILE = '/data_categorical.csv'
K = 5

data = pd.read_csv(FOLDER + FILE, index_col=False)

# Splitting the data into train and test sets (80% train, 20% test)
train_valid_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

for df in [train_valid_data, test_data]:
    df.reset_index(drop=True, inplace=True)

X_train_valid, y_train_valid = train_valid_data.iloc[:, :-1], train_valid_data.iloc[:, -1]
X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
val_accuracies = []
for fold_idx, fold in enumerate(skf.split(X_train_valid, y_train_valid)):
    train_index, val_index = fold
    X_train_fold, X_val_fold = X_train_valid.iloc[train_index], X_train_valid.iloc[val_index]
    y_train_fold, y_val_fold = y_train_valid[train_index], y_train_valid[val_index]

    fold_predictor = oner.OneR()
    fold_predictor.fit(X_train_fold, y_train_fold)

    preds = fold_predictor.predict(X_val_fold)
    mean_acc = accuracy_score(preds, y_val_fold.to_list())
    val_accuracies.append(mean_acc)

kf_scores = {'test_score': np.array(val_accuracies)}
dd.visualize_cv(K, kf_scores, FOLDER, f'oner_')

clf = oner.OneR()
clf.fit(X_train_valid, y_train_valid)
print("Best predictor: ", clf.best_predictor)

y_pred = clf.predict(X_test)
dd.visualize_cr_cm(y_test.to_list(), y_pred, FOLDER, f'oner_')
