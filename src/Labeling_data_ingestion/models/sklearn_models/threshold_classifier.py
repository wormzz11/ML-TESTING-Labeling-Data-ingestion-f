from sklearn.base import BaseEstimator, ClassifierMixin

class ThresholdClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, threshold=0.5):

        self.estimator = estimator
        self.threshold = threshold
    
    def fit(self, X, y):

        self.estimator.fit(X, y)
        self.classes_ = self.estimator.classes_
        return self
    
    def predict(self, X):

        scores = self.estimator.predict_proba(X)[:,1]
        return (scores >= self.threshold).astype(int)
        
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)    
        
        