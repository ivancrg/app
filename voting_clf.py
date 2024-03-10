import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from vc.dt.main import DecisionTreeVC
from vc.seqcov.main import SequentialCoveringVC
from vc.rf.main import RandomForestVC
from vc.ml_p.main import MLPVC
from sklearn.metrics import confusion_matrix, classification_report

output_folder = './report_test/histology_binary_test'

data = pd.read_csv(f'{output_folder}/data_norm.csv')
labels, X_labels, y_labels = data.columns.to_list(), data.columns.to_list()[:-1], [data.columns.to_list()[-1]]
X, y = data.iloc[:, :-1], data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_labels, y_labels)
print(data.head())

X_train, X_test, y_train, y_test = X_train.to_numpy(), X_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()

clf1 = MLPVC(output_folder + '/MLP', smote=False, n_splits=5, verbose=True)
clf2 = RandomForestVC(output_folder + '/RF')
clf3 = SequentialCoveringVC(X_labels, y_labels, output_folder + '/SEQCOV')
clf4 = DecisionTreeVC(output_folder + '/DT')

vc = VotingClassifier(estimators=[
        ('mlp', clf1), ('rf', clf2), ('seqcov', clf3), ('dt', clf4)], voting='hard')
vc = vc.fit(X_train, y_train)

y_hat = vc.predict(X_test)
print(confusion_matrix(y_test, y_hat))
print(classification_report(y_test, y_hat))
