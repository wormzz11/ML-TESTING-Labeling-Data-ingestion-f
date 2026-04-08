from Labeling_data_ingestion.config import DATA_PATH
from  Labeling_data_ingestion.data_handler.import_data import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = load_data(DATA_PATH)
df = df[["title", "theme", "relevant"]]

def train(model, threshold = 0.348):
    

    X = df["title"] + " " + df["theme"] 
    y = df["relevant"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40, test_size=0.20)
    
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2),sublinear_tf=True)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    model.fit(X_train_vect, y_train)

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test_vect)[:, 1]
    else:
        scores = model.decision_function(X_test_vect)

    
   

    y_pred = (scores >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    print(classification_report(y_test, y_pred))
    
    return model, vectorizer, accuracy, scores
