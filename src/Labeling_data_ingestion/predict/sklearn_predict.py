import pandas as pd
import joblib
from Labeling_data_ingestion.data_handler.import_data import load_data 

df = load_data(r"data/testing_data.csv")
df = df.dropna(subset=['theme'])



pipeline = joblib.load("trained_models/pipeline.rfk")
X = df["title"] + " " + df["theme"]

scores = pd.Series(pipeline.predict_proba(X)[:, 1], index=df.index)

certain_relevant = scores >= 0.6
certain_irrelevant = scores > 0.3
manual_review = ~certain_irrelevant & ~certain_relevant

df.loc[certain_irrelevant, "relevant"] = 1.0
df.loc[certain_irrelevant, "relevant"] = 0.0
df.loc[manual_review, "relevant"] = None

df.loc[certain_irrelevant | certain_relevant].to_csv("data/certain_auto/auto_labeled.csv", index=False)
df.loc[manual_review].to_csv("data/manual_review/manual_review.csv", index=False)
















