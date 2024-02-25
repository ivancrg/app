import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from vc.dt.main import DecisionTreeVC
from vc.seqcov.main import SequentialCoveringVC
from vc.rf.main import RandomForestVC
from vc.ml_p.main_gs import MLPVC

data = pd.read_csv('./report/SEV/data_norm.csv')
labels, X_labels, y_labels = data.columns.to_list(), data.columns.to_list()[:-1], [data.columns.to_list()[-1]]
X, y = data.iloc[:, :-1], data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_labels, y_labels)
print(data.head())

X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

clf1 = DecisionTreeVC('./report/voting_test')
clf2 = SequentialCoveringVC(X_labels, y_labels, './report/voting_test')
clf3 = RandomForestVC('./report/voting_test')
clf4 = MLPVC('./report/voting_test')

vc = VotingClassifier(estimators=[
        ('dt', clf1), ('seqcov', clf2), ('rf', clf3)], voting='hard')
vc = vc.fit(X, y)

y_hat = vc.predict(X)
