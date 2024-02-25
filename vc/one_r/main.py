import pandas as pd
from one_r.oner import OneR
from sklearn.model_selection import train_test_split
import display_data as dd


class OneRule():
    def __init__(self, folder, file):
        self.folder = folder
        self.file = file

    def run(self):
        data = pd.read_csv(self.folder + self.file, index_col=False)

        # Splitting the data into train and test sets (80% train, 20% test)
        train_data, test_data = train_test_split(
            data, test_size=0.2, random_state=42, shuffle=True)

        for df in [train_data, test_data]:
            df.reset_index(drop=True, inplace=True)

        X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
        X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

        clf = OneR()
        clf.fit(X_train, y_train)
        print("Best predictor: ", clf.best_predictor)

        y_pred = clf.predict(X_test)
        dd.visualize_cr_cm(y_test.to_list(), y_pred, self.folder, f'oner_')
