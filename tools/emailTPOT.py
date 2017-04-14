from __future__ import print_function
import sys
sys.path.append("../tools/")
from email_preprocess import preprocess
from tpot import TPOTClassifier

features_train, features_test, labels_train, labels_test = preprocess()

tpot = TPOTClassifier()
tpot.fit(features_train, labels_train)
print(tpot.score(features_test, labels_test))
tpot.export('tpot_email_pipeline.py')