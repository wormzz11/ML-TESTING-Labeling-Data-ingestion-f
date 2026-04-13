import pandas as pd
import joblib
from Labeling_data_ingestion.data_handler.process_data import load_data, append_csv


df = load_data(r"data/predict_data/unlabeled.csv")
print("Missing theme count: " + str((df['theme'].isna().sum())))
df = df.dropna(subset=['theme'])



pipeline = joblib.load("trained_models/transformer_pipeline.rfk")
X = df["title"] + " " + df["theme"]

scores = pd.Series(pipeline.predict_proba(X)[:, 1], index=df.index)

#change for manual review
certain_relevant = scores >= 0.25
certain_irrelevant = scores < 0.25

manual_review = ~certain_irrelevant & ~certain_relevant

df.loc[certain_relevant, "relevant"] = 1.0
df.loc[certain_irrelevant, "relevant"] = 0.0
df.loc[manual_review, "relevant"] = None



append_csv(df.loc[certain_irrelevant | certain_relevant], "data/certain_auto/auto_labeled_test.csv")
append_csv(df.loc[manual_review], "data/manual_review/manual_review.csv")

manual_df = df.loc[manual_review].copy()
manual_df["score"] = scores[manual_review]
print(manual_df[["title", "theme", "score"]].sort_values("score", ascending=False).head(20))
print(manual_df[["title", "theme", "score"]].sort_values("score").head(20))










