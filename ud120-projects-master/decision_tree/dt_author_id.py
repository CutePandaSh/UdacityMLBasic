#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
print len(features_train[0])

from sklearn.tree import DecisionTreeClassifier as DTC
clf = DTC(min_samples_split=40)
print "start fitting...."
t0 = time()
clf.fit(features_train, labels_train)
print 'training time: ',time()-t0

print 'start predicting...'
t0 = time()
pred = clf.predict(features_test)
print 'predict time: ', time()-t0


print accuracy_score(pred, labels_test)


#########################################################
### your code goes here ###


#########################################################


