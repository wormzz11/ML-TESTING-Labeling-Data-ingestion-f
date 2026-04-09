from Labeling_data_ingestion.models.sklearn_models.sk_models import logistic_model
from Labeling_data_ingestion.train.train_sklearn import train

pipe, accuracy = train(logistic_model(), threshold=0.3)
    
