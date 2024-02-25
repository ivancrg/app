import pandas as pd

class OneR():
    def __init__(self):
        self.best_predictor = None

    def calculate_error(self, predictor, result, value):
        # print(f'{predictor.name} == {value}')
        
        # How often does each result appear for the given value of the predictor
        value_counts = result[predictor == value].value_counts()
        # print(value_counts)

        # The most frequent class in relation to this predictor's value
        # (The class (result) that is most commonly seen in combination with this predictor's value)
        most_frequent_class = value_counts.idxmax()

        # Calculate the total error (misclassifications) for this value of the predictor
        error = len(result[predictor == value]) - value_counts[most_frequent_class]
        # print(f'error = {len(result[predictor == value])} - {value_counts[most_frequent_class]} = {error}')

        return value, error, most_frequent_class


    def fit(self, X, y):
        # Initial parameter setup
        best_error = float('inf')

        # Try all classes as a feature
        for feature in X.columns:
            # Find the total error of the selected feature for each possible feature value
            total = [self.calculate_error(X[feature], y, value)
                    for value in X[feature].unique()]
            
            print(f'Prediction error for "{feature}" feature: {total}')

            # With optimum setup, how much misclassifications will a predictor based on this feature have?
            total_error = sum(e for _, e, _ in total)

            if total_error < best_error:
                self.best_predictor = {
                    'feature': feature,
                    'values': [v for v, _, _ in total],
                    'result': [r for _, _, r in total]
                }

                best_error = total_error

            print(
                f'{feature} predictor accuracy: {round((len(X) - total_error) / len(X), 3)}')


    def predict(self, X):
        preds = []

        for _, row in X.iterrows():
            for i, value in enumerate(self.best_predictor['values']):
                if row[self.best_predictor['feature']] == value:
                    preds.append(self.best_predictor['result'][i])
        
        return preds