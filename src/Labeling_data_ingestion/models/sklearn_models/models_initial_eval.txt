
#Best performances achieved by each model for current purpose of loose sieve as an initial step in data labeling with TF-IDF
#Logistic regression

#[[122 195]
# [ 17 131]]
#target: 0,8<= recall, while precision>40 for 0.3 threshold 

#              precision    recall  f1-score   support

#           0       0.88      0.38      0.54       317
#           1       0.40      0.89      0.55       148

#    accuracy                           0.54       465
#   macro avg       0.64      0.63      0.54       465
#weighted avg       0.73      0.54      0.54       465

#0.5440860215053763


#MultiNomialNB_model for 0.03 threshold

#[[146 171]
# [ 33 115]]


#              precision    recall  f1-score   support

#           0       0.82      0.46      0.59       317
#           1       0.40      0.78      0.53       148

#    accuracy                           0.56       465
#   macro avg       0.61      0.62      0.56       465
#weighted avg       0.68      0.56      0.57       465

#0.5612903225806452



#ComplementNB_model 0.33 threshold

#[[139 178]
# [ 30 118]]


#              precision    recall  f1-score   support

#           0       0.82      0.44      0.57       317
#           1       0.40      0.80      0.53       148

#    accuracy                           0.55       465
#   macro avg       0.61      0.62      0.55       465
#weighted avg       0.69      0.55      0.56       465

#0.5526881720430108


#SGDClassifier for 0.25 threshold 

#[[123 194]
# [ 18 130]]


#              precision    recall  f1-score   support

#           0       0.87      0.39      0.54       317
#           1       0.40      0.88      0.55       148

#    accuracy                           0.54       465
#   macro avg       0.64      0.63      0.54       465
#weighted avg       0.72      0.54      0.54       465

#0.5440860215053763


#SGDClassifier_Hinge, not usable for current data, below is the output from old dataset 


#              precision    recall  f1-score   support

#           0       0.90      0.80      0.84       173
#           1       0.65      0.80      0.72        82

#    accuracy                           0.80       255
#   macro avg       0.77      0.80      0.78       255
#weighted avg       0.82      0.80      0.80       255

#0.8


#linear svc, no matter the threshold or increasing the bias it stays skewed towards 0 so its not usable for the purpose of a loose sieve with high recall for
#[[267  50]
# [ 61  87]]


#              precision    recall  f1-score   support

#           0       0.81      0.84      0.83       317
#           1       0.64      0.59      0.61       148

#    accuracy                           0.76       465
#   macro avg       0.72      0.72      0.72       465
#weighted avg       0.76      0.76      0.76       465

#0.7612903225806451


#Overall logistic regression has the best performance for the purpose of high recall lose sieve with close second log loss sgd