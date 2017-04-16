#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL' , 0)
data = featureFormat(data_dict, features)

for point in data:
	salary = point[0]
	bonus = point[1]
	plt.scatter(salary, bonus)

plt.xlabel('Salary')
plt.ylabel('Bonus')
plt.show()

### your code below



