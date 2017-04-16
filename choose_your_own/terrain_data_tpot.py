from __future__ import print_function
from tpot import TPOTClassifier
from prep_terrain_data import makeTerrainData
import numpy as np

features_train, labels_train, features_test, labels_test = makeTerrainData()
features_train = np.array(features_train)
labels_train = np.array(labels_train)
features_test = np.array(features_test)
labels_test = np.array(labels_test)

tpot = TPOTClassifier(generations = 2 ,verbosity=2)
tpot.fit(features_train, labels_train)
print(tpot.score(features_test, labels_test))
tpot.export('terrain_data_pipeline.py')