import pandas as pd
from sequential_covering import SequentialCovering
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import display_data as dd

FOLDER = './report/ND/p2'
MULTICLASS = False
FILE = '/data.csv'
K = 5

def cv(X_train_valid, y_train_valid, params):
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    val_accuracies = []
    for fold_idx, fold in enumerate(skf.split(X_train_valid, y_train_valid)):
        train_index, val_index = fold
        X_train_fold, X_val_fold = X_train_valid.iloc[train_index], X_train_valid.iloc[val_index]
        y_train_fold, y_val_fold = y_train_valid[train_index], y_train_valid[val_index]
        
        train_data = pd.concat([X_train_fold, pd.DataFrame(y_train_fold)], axis=1, ignore_index=True).reset_index().iloc[:, 1:]
        train_data.columns = data.columns

        valid_data = pd.concat([X_val_fold, pd.DataFrame(y_val_fold)], axis=1, ignore_index=True).reset_index().iloc[:, 1:]
        valid_data.columns = data.columns

        sc = SequentialCovering(
            train_data,
            multiclass=MULTICLASS,
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            max_leaf_nodes=params['max_leaf_nodes'],
            output_name=PREDICTION_LABEL)
        sc.fit()

        preds = sc.predict_tmp(valid_data)
        mean_acc = (preds[PREDICTION_LABEL] == preds['Prediction']).mean()
        val_accuracies.append(mean_acc)
        
    kf_scores = {'test_score': np.array(val_accuracies)}

    return (np.mean(kf_scores['test_score']), kf_scores)

data = pd.read_csv(FOLDER + FILE, index_col=False)

PREDICTION_LABEL = data.columns[-1]

# Splitting the data into train and test sets (80% train, 20% test)
train_valid_data, test_data = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)

for df in [train_valid_data, test_data]:
    df.reset_index(drop=True, inplace=True)

X_train_valid, y_train_valid = train_valid_data.iloc[:, :-1], train_valid_data.iloc[:, -1]

best_mean_acc, best_kf_scores, best_params = 0, {}, {}
param_grid = {
    'max_leaf_nodes': range(10, 31, 5),
    'max_features': range(2, 12, 3),
    'max_depth': range(3, 15, 3),
    'min_samples_split': range(2, 10, 3),
    'min_samples_leaf': range(1, 12, 5)
}

for max_leaf_nodes in param_grid['max_leaf_nodes']:
    for max_features in param_grid['max_features']:
        for max_depth in param_grid['max_depth']:
            for min_samples_split in param_grid['min_samples_split']:
                for min_samples_leaf in param_grid['min_samples_leaf']:
                    cur_gs_params = {
                        'max_leaf_nodes': max_leaf_nodes,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf
                    }

                    mean_acc, kf_scores = cv(
                        X_train_valid=X_train_valid.copy(),
                        y_train_valid=y_train_valid.copy(),
                        params=cur_gs_params
                        )
                    
                    if mean_acc > best_mean_acc:
                        best_mean_acc, best_kf_scores, best_params = mean_acc, kf_scores, cur_gs_params
                    
dd.visualize_cv(K, kf_scores, FOLDER, f'seqcov_best_')
with open(FOLDER + f'/seqcov_best_params.txt', 'w') as file:
    for key, value in best_params.items():
        file.write(f'{key}: {value}\n')

# TEST
sc = SequentialCovering(train_valid_data, multiclass=MULTICLASS, max_depth=7, min_samples_leaf=2, output_name=PREDICTION_LABEL)
sc.fit()

print("Learned Rules:")
sc.print_rules_preds()

preds = sc.predict_tmp(test_data)
y_true, y_pred = [tc[0] for tc in pd.DataFrame(preds.iloc[:, -2]).to_numpy()], [pc[0] for pc in pd.DataFrame(preds.iloc[:, -1]).to_numpy()]
dd.visualize_cr_cm(y_true, y_pred, FOLDER, f'seqcov_best_')


import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

plt.figure(figsize=(12, 10))

results = permutation_importance(
    sc,
    X_train_valid,
    y_train_valid,
    n_repeats=10,
    random_state=42
    )

X_train_valid = X_train_valid.drop('Prediction', axis=1)
importance = results.importances_mean[:-1] # Drop 'Prediction' importance

for i, v in enumerate(importance):
    print(f'Feature {X_train_valid.columns[i]}: {v:.5f}')

plt.subplots_adjust(left=0.09, right=0.96, bottom=0.237, top=0.97)
plt.bar(X_train_valid.columns, importance)
plt.xticks(rotation=90)
plt.ylabel('Importance')
plt.savefig(FOLDER + '/seqcov_best_fimp.png')