#!/usr/bin/python
from __future__ import print_function
import matplotlib.pyplot as plt
from time import time
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
number_of_neighbors = input('Enter the number of nearest neighbors: ')
clf = KNeighborsClassifier(n_neighbors=number_of_neighbors, weights='uniform', p = 2, metric='minkowski')
start_time = time()
clf.fit (features_train, labels_train)
print('Training time = {:.4f}'.format(time()-start_time))

start_time = time()
pred = clf.predict(features_test)
print('Prediction time = {:.4f}'.format(time()-start_time))

accuracy = accuracy_score(labels_test, pred)
print('Accuracy = {:.2f} % with {:d} nearest neighbors '.format(100 * accuracy, number_of_neighbors))


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
