import joblib
from Labeling_data_ingestion.data_handler.import_data import load_data 
df = load_data(r"data\unlabeled.csv")
print(df["theme"].isna().sum())
df = df.dropna(subset=['theme'])
print(df["theme"].isna().sum())

pipeline = joblib.load("trained_models/pipeline.rfk")

replace = df["relevant"].isna()
X = df.loc[replace, "title"] + " " + df.loc[replace, "theme"]
df.loc[replace, "relevant"] = pipeline.predict(X)


print(f"Labeled {replace.sum()} rows")
print(df["relevant"].value_counts())

df.to_csv("data/data_mlLabeled.csv", index=False)




