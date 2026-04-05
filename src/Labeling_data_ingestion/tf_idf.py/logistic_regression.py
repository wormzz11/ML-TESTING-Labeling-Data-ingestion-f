import pandas as pd
from Labeling_data_ingestion.config import DATA_PATH
from  Labeling_data_ingestion.data_handler.import_data import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = load_data(DATA_PATH)
df = df[["title", "theme", "relevant"]]

def asses_relevance(df):
    corpus = df["title"] + " " + df["theme"]
    Tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    x = Tfidf_vectorizer.fit_transform(corpus)


    
