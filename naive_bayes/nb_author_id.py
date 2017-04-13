#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
from __future__ import print_function  
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
clf = GaussianNB() # Create the classifier

# Time the training of the classifier
start_time = time()
clf.fit(features_train, labels_train) # Train on the testing features and labels
print('Training time = {:.4f}'.format(time()-start_time))

start_time = time()
pred = clf.predict(features_test)
print('Prediction time = {:.4f}'.format(time()-start_time))

accuracy = accuracy_score(labels_test, pred)

print('Accuracy = {:.4f}'.format(accuracy))



#########################################################
### your code goes here ###


#########################################################


