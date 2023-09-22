import pandas as pd
from sequential_covering import SequentialCovering
from sklearn.model_selection import train_test_split

data = pd.read_csv('./final/postop_spl_cplx_othr.csv', index_col=False)


# Splitting the data into train and test sets (80% train, 20% test)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

for df in [train_data, test_data]:
    df.reset_index(drop=True, inplace=True)
    
sc = SequentialCovering(train_data, True, max_depth=7, min_samples_leaf=2, output_name='Postoperative Diagnosis')
sc.fit()

print("Learned Rules:")
sc.print_rules_preds()

preds = sc.predict_tmp(test_data)
print((preds['Postoperative Diagnosis'] == preds['Prediction']).mean())
print(preds['Prediction'].value_counts())
y_test_class, y_pred_class = [tc[0] for tc in pd.DataFrame(preds.iloc[:, -2]).to_numpy()], [pc[0] for pc in pd.DataFrame(preds.iloc[:, -1]).to_numpy()]


from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test_class, y_pred_class))

from sklearn.metrics import classification_report
print(classification_report(y_test_class, y_pred_class))