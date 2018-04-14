#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
    Sara has label 0
    Chris has label 1
"""

import sys
import numpy
from time import time
from sklearn import svm
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing datasets, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################

# Use online a subset of the data?
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

# Use different Cs?
# Cs = [10, 100, 1000, 10000]
Cs = [10000]

for C in Cs:
    print "C = ", C
    clf = svm.SVC(kernel="rbf", C=C)

    t = time()
    clf.fit(features_train, labels_train)
    print "\ttraining time:", round(time()-t, 3), "s"

    t = time()
    predicted = clf.predict(features_test)
    print "\tpredict time:", round(time()-t, 3), "s"

    print "\t10: ", predicted[10]
    print "\t26: ", predicted[26]
    print "\t50: ", predicted[50]

    _, count = numpy.unique(predicted, return_counts=True)

    print "\tNumber Chris entries: ", count[1]

    print "\tscore: ", clf.score(features_test, labels_test)
#########################################################
