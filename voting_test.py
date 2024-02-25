from sklearn.base import BaseEstimator, ClassifierMixin

class VotingTest(BaseEstimator, ClassifierMixin):
    def __init__(self, name):
        self.name = name
        print(f'VotingTest {name} initialized.')
    
    def fit(self, X, y):
        print(f'Fitting VotingTestClassifier {self.name}\n{X}\n{y}\n\n')

    def predict(self, X):        
        print(f'Predicting VotingTestClassifier {self.name}\n{X}\n\n')