from sklearn.linear_model import LogisticRegression, SGDClassifier
from Labeling_data_ingestion.train.train_sklearn import train
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes  import ComplementNB
from sklearn.svm import LinearSVC



def logistic_model():
    
    return  LogisticRegression(
    l1_ratio=0,
    C=13.0,
    solver="saga",
    max_iter=1000,
    class_weight="balanced"
    )



def MultiNomialNB_model():
    
    return MultinomialNB(
    alpha = 1
    )



def ComplementNB_model():

    return ComplementNB(alpha=0.5)



def SGDClassifier_model():

    return SGDClassifier(loss="log_loss",
    class_weight="balanced",
    max_iter=1000,
    tol=1e-3,
    alpha=0.00001
    )

result = train(ComplementNB_model())
print(result[2])


#Logistic regression

#[[111  62]
# [ 14  68]]


#              precision    recall  f1-score   support

#           0       0.89      0.64      0.74       173
#           1       0.52      0.83      0.64        82

#    accuracy                           0.70       255
#   macro avg       0.71      0.74      0.69       255
#weighted avg       0.77      0.70      0.71       255

# acc 0.7019607843137254



#MultiNomialNB_model

#[[125  48]
 #[ 21  61]]


#              precision    recall  f1-score   support

#           0       0.86      0.72      0.78       173
#           1       0.56      0.74      0.64        82

#    accuracy                           0.73       255
#   macro avg       0.71      0.73      0.71       255
#weighted avg       0.76      0.73      0.74       255

#acc 0.7294117647058823



#ComplementNB_model

#[[85 88]
#[13 69]]
#              precision    recall  f1-score   support

#          0       0.87      0.49      0.63       173
#           1       0.44      0.84      0.58        82

#    accuracy                           0.60       255
#   macro avg       0.65      0.67      0.60       255
#weighted avg       0.73      0.60      0.61       255

#acc 0.6039215686274509



#SGDClassifier

#[[131  42]
 #[ 16  66]]


#              precision    recall  f1-score   support

#           0       0.89      0.76      0.82       173
#           1       0.61      0.80      0.69        82

#    accuracy                           0.77       255
#   macro avg       0.75      0.78      0.76       255
#weighted avg       0.80      0.77      0.78       255

#0.7725490196078432