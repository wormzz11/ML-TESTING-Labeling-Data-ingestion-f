import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from Labeling_data_ingestion.data_handler.process_data import load_data, append_csv, build_prediction_dataset
from Labeling_data_ingestion.config import ThresholdConfig
 
 
def run_prediction(input_path, model_path):
 
    config    = ThresholdConfig()
    df        = load_data(input_path)
    df        = df.dropna(subset=["theme"]).copy()
    pipe      = joblib.load(model_path)
    X         = build_prediction_dataset(df)
    scores    = pd.Series(pipe.predict_proba(X)[:, 1], index=df.index)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    certain_relevant   = scores >= config.high_pos
    certain_irrelevant = scores <= config.high_neg
    manual_review      = ~certain_relevant & ~certain_irrelevant
 
    df.loc[certain_relevant,   "relevant"] = 1.0
    df.loc[certain_irrelevant, "relevant"] = 0.0
    df.loc[manual_review,      "relevant"] = None
    df["score_t1"] = scores
 
    certain_df = df.loc[certain_relevant | certain_irrelevant].copy()
    manual_df  = df.loc[manual_review].copy()
 
    batch_certain_path = f"data/certain_auto/batches/batch_{timestamp}.csv"
    all_certain_path   = "data/certain_auto/auto_labeled_all.csv"
    manual_path        = f"data/manual_review/manual_{timestamp}.csv"
 
    Path(batch_certain_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manual_path).parent.mkdir(parents=True, exist_ok=True)
 
    certain_df.to_csv(batch_certain_path, index=False)
 
    append_csv(certain_df, all_certain_path)
 
    all_df = pd.read_csv(all_certain_path)
    all_df.drop_duplicates(subset=["title"], keep="last", inplace=True)
    all_df.to_csv(all_certain_path, index=False)
 
    manual_df.to_csv(manual_path, index=False)
 
    print(f"Certain relevant:   {certain_relevant.sum()}")
    print(f"Certain irrelevant: {certain_irrelevant.sum()}")
    print(f"Manual review:      {manual_review.sum()}")
    print(f"Batch saved  → {batch_certain_path}")
    print(f"All updated  → {all_certain_path}")
    print(f"Manual saved → {manual_path}")
 
 
if __name__ == "__main__":
    run_prediction(
        input_path="data/predict_data/unlabeled.csv",
        model_path="trained_models/transformer_pipeline.rfk"
    )

