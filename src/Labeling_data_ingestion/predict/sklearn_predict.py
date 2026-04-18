import pandas as pd
import joblib
from Labeling_data_ingestion.data_handler.process_data import load_data, append_csv, build_prediction_dataset
from Labeling_data_ingestion.config import ThresholdConfig

def run_prediction(input_path, model_path):

    config = ThresholdConfig()
    df = load_data(input_path)
    df = df.dropna(subset=['theme'])
    df = df.copy()
    pipe = joblib.load(model_path)
    
    X = build_prediction_dataset(df)

    scores = pd.Series(pipe.predict_proba(X)[:, 1], index=df.index)

    certain_relevant = scores >= config.high_pos
    certain_irrelevant = scores <= config.high_neg
    manual_review = (scores > config.high_neg) & (scores < config.high_pos)


    df.loc[certain_relevant, "relevant"] = 1.0
    df.loc[certain_irrelevant, "relevant"] = 0.0
    df.loc[manual_review, "relevant"] = None
    df['score_t1'] = scores


    append_csv(df.loc[certain_irrelevant | certain_relevant], "data/certain_auto/auto_labeled_pipeline_test2.csv")
    append_csv(df.loc[manual_review], "data/manual_review/manual_review_test2.csv")

    manual_df = df.loc[manual_review].copy()
    manual_df["score"] = scores[manual_review]






if __name__ == "__main__":
    run_prediction(
        input_path="data/predict_data/unlabeled.csv",
        model_path="trained_models/transformer_pipeline.rfk"
    )





