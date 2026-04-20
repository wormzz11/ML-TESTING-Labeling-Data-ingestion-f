from sklearn.linear_model import LogisticRegression #, SGDClassifier
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.naive_bayes  import ComplementNB
# from sklearn.svm import LinearSVC



def logistic_model():
    
    return  LogisticRegression(
    l1_ratio=0,
    C=0.13,
    solver="saga",
    max_iter=1000,
    random_state=42
    )



# def MultiNomialNB_model():
    
#     return MultinomialNB(
#     alpha = 0.00005
#     )



# def ComplementNB_model():

#     return ComplementNB(alpha=1)



# def SGDClassifier_model():

#     return SGDClassifier(loss="log_loss",
#     max_iter=1000,
#     tol=1e-3,
#     alpha=0.001,
#     random_state=42
#     )

# #deprecated
# def linearSVC_model():

#     return LinearSVC(
#     penalty="l2",
#     loss="squared_hinge",
#     dual=False,
#     tol=1e-3,
#     C=100,
#     class_weight="balanced",
#     max_iter=1000,
#     random_state=42
#     )
# #deprecated 
# def SGDClassifier_hinge():
#     return SGDClassifier(
#     loss ="hinge",  
#     alpha=0.001,
#     max_iter=1000,
#     random_state=42
#     )




