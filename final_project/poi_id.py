#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import pandas as pd

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Create a dataframe from the dictionary
df = pd.DataFrame(data_dict)

# Transpose so the rows are individuals and columns are data fields
df = df.transpose()
df.drop(['TOTAL'], inplace=True)

# Create a list of all the data fields
fields = df.columns.tolist()
# All data fields are numeric except email_address and poi
fields.remove('email_address')
fields.remove('poi')

# Iterate through all the columns and convert data to floating point
for field in fields:
	try:
		df[field] = df[field].astype(float)
	except Exception as e:
		print e

# Create new features
# First feature is ratio of emails recieved that are from a person of interest
df['from_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
# Second new feature is ratio of emails sent that are to a person of interest
df['to_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
# Third new feature is ratio of emails received that shared a receipt with a person of interest
df['shared_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']

# Drop all NaNs from data
df = df.dropna(subset=['to_ratio', 'from_ratio', 'shared_ratio'])

# Transpose back and convert to dictionary where each key is the individual and values are dictionaries of the data
data_dict = df.transpose().to_dict()

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'to_ratio', 'from_ratio'] 

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



clf = DecisionTreeClassifier()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)