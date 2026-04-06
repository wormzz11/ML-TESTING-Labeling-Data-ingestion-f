from Labeling_data_ingestion.config import DATA_PATH
from  Labeling_data_ingestion.data_handler.import_data import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score

df = load_data(DATA_PATH)
df = df[["title", "theme", "relevant"]]

def train(model):
    X = df["title"] + " " + df["theme"]
    y = df["relevant"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40, test_size=0.15)
    
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2),sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, vectorizer, accuracy
