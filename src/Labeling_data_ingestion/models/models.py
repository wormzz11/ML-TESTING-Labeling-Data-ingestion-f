from sklearn.linear_model import LogisticRegression
from Labeling_data_ingestion.train.train import train
def logistic_model():
    
    return  LogisticRegression(
    l1_ratio=0,
    C=13.0,
    solver="saga",
    max_iter=1000,
    class_weight="balanced"
    )


result = train(logistic_model())
print(result[2])




    
