from Labeling_data_ingestion.config import DATA_PATH
from  Labeling_data_ingestion.data_handler.process_data import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from  sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.pipeline import Pipeline
from Labeling_data_ingestion.models.sklearn_models.threshold_classifier import ThresholdClassifier


def train(model, threshold = 0.3):
    
    df = load_data(DATA_PATH)
    df = df[["title", "theme", "relevant"]].dropna()
    

    X = df["title"] + " " + df["theme"] 
    y = df["relevant"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40, test_size=0.20)
    
    wrapped = ThresholdClassifier(model, threshold=threshold)

    pipe = Pipeline([(
        "vectorizer", 
        TfidfVectorizer(
            stop_words="english", 
            ngram_range=(1,2),
            sublinear_tf=True
            )),
            ("model", wrapped)
    ])


    pipe.fit(X_train, y_train)



    y_pred = pipe.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(classification_report(y_test, y_pred))

    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipe, "trained_models/pipeline.rfk")

    return pipe, accuracy
