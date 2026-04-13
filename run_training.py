from Labeling_data_ingestion.models.sklearn_models.sk_models import logistic_model
from Labeling_data_ingestion.train.train import train_tfidf, train_transformer


#threshold is for testing performance before running prediction
pipe, accuracy = train_transformer(logistic_model(), threshold=0.25)
    
