from sentence_transformers import SentenceTransformer
from sklearn.base import TransformerMixin, BaseEstimator


class MiniLmVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.model.encode(X.tolist(), show_progress_bar=True)

        