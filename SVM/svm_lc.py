# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 16:18:03 2016

@author: lancel
"""


# fit svm model with linear kernael
from sklearn.svm import SVC

svm_classifier = SVC(kernel = 'linear', random_state = 0)
svm_classifier.fit(X_train, y_train)

# predict results 
y_pred_ln = svm_classifier.predict(X_test)



