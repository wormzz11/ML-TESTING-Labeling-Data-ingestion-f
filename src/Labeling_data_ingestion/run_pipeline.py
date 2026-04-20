from Labeling_data_ingestion.config import DATA_PATH, ThresholdConfig
from Labeling_data_ingestion.data_handler.process_data import build_dataset, load_data
from Labeling_data_ingestion.models.sklearn_models.sk_models import logistic_model
from Labeling_data_ingestion.train.train import (
    train_test,
    train_tfidf,
    train_transformer,
    evaluate,
    save_model,
)

models = {
    "tfidf":       train_tfidf,
    "transformer": train_transformer,
}


def run_pipeline(model_name="transformer"):
    config = ThresholdConfig()
    df     = load_data(DATA_PATH)
    df     = build_dataset(df)

    X_train, X_test, y_train, y_test = train_test(df, 0.2)

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    base_model = logistic_model()
    pipe       = models[model_name](base_model, X_train, y_train)
    results    = evaluate(pipe, X_test, y_test, config.high_pos)
    save_model(pipe, f"trained_models/{model_name}.rfk")

    return pipe, results


if __name__ == "__main__":
    run_pipeline("transformer")