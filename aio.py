from dt.main import DecisionTreeCV
from one_r.main import OneRule
from seqcov.main_cv import SeqCovCV
from rf.main import RandomForestCV
from ml_p.main import MLPCV
import pandas as pd
from sklearn.model_selection import train_test_split
from vc.ml_p.main import MLPVC

data = pd.read_csv('./report_test/SEV/data_norm.csv')
labels, X_labels, y_labels = data.columns.to_list(), data.columns.to_list()[:-1], [data.columns.to_list()[-1]]
X, y = data.iloc[:, :-1], data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_labels, y_labels)
print(data.head())

X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


# Evaluates all algorithms - GS+CV to get the best parameters
# and scoring on test set using the calculated parameters (except OneR)

folder, file, file_norm, file_cat = './report_test/SEV', '/data.csv', '/data_norm.csv', '/data_cat.csv'

# print('Decision tree running...')
# dt = DecisionTreeCV(folder, file)
# dt.run()

# print('OneRule running...')
# oner = OneRule(folder, file_cat)
# oner.run()

# print('Sequential covering running...')
# sc = SeqCovCV(folder, file)
# sc.run()

# print('Random forest running...')
# ranf = RandomForestCV(folder, file)
# ranf.run()

print('MLP running...')
mlp = MLPVC('./report_test/voting_test')
mlp.fit(X, y)