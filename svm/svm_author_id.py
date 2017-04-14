#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
from __future__ import print_function
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
features_train = features_train[:len(features_train)/100] # cut size of training features by factor of 100
labels_train = labels_train[:len(labels_train)/100] # cut size of training labels by factor of 100

clf = SVC(kernel = 'rbf', C=10000) 
start_time = time()
clf.fit(features_train, labels_train)
print('Training time = {:.2f}'.format(time()-start_time))

start_time = time()
pred = clf.predict(features_test)
print('Prediction time = {:.2f}'.format(time()-start_time))

accuracy = accuracy_score(labels_test, pred)
print('Accuracy = {:.2f}%%'.format(100*accuracy))


#########################################################
### your code goes here ###

#########################################################


