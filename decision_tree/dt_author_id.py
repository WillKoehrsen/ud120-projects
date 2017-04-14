#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
from __future__ import print_function 
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

min_samples_split = 40
clf = tree.DecisionTreeClassifier(min_samples_split=40) # initialize the decision tree classifier with min_samples_split of 40, node will not be split unless it contains at least 40 samples
start_time = time()
clf.fit(features_train, labels_train) # train the classifier
print('Training time = {:.2f}'.format(time()-start_time))

pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test, pred)
print('Prediction time = {:.2f}'.format(time()-start_time))

print('Accuracy with min_samples_split = {:d} is {:.2f} %'.format(min_samples_split, 100*accuracy))

#########################################################
### your code goes here ###


#########################################################


