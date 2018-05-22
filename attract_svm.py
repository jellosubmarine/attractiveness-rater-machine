#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

STDDEV_THRESHOLD = 1.0

def read_in_feature_vectors(path):
    fin = open(path, 'r')
    t = [ line.split(' ') for line in fin.read().split('\n') if line ]
    fin.close()
    fvs = []
    for line in t:
        fvs.append([ float(x) for x in line ])
    return fvs

def read_in_ratings(path):
    fin = open(path,'r')
    t = [ line.split(',') for line in fin.read().split('\n') if line ][1:]
    fin.close()
    ratings = [ float(line[1].strip()) for line in t ]
    return ratings

def read_in_stddev_info(path):
    fin = open(path,'r')
    t = [ line.split(',') for line in fin.read().split('\n') if line ][1:]
    fin.close()
    stddevs = [ float(line[2].strip()) for line in t ]
    return stddevs

def divide_dataset(vectorset, labels, part_size):
    datalen = len(vectorset)
    shuffled_range = list(range(datalen))
    random.shuffle(shuffled_range)
    trainset = []
    trainlabels = []
    testset = []
    testlabels = []
    for i in shuffled_range[:int(datalen*part_size)]:
        trainset.append(copy.deepcopy(vectorset[i]))
        trainlabels.append(labels[i])
    for i in shuffled_range[int(datalen*part_size):]:
        testset.append(copy.deepcopy(vectorset[i]))
        testlabels.append(labels[i])
    return trainset, trainlabels, testset, testlabels

feature_vecs = read_in_feature_vectors("featurevectors.txt")
ratings = read_in_ratings("rating_withoutfail.csv")
stddevs = read_in_stddev_info("rating_withoutfail.csv")

trainset, trainlabels, testset, testlabels = divide_dataset(feature_vecs, ratings, 0.9)

#~ classifier = SVR()
#~ classifier = RandomForestRegressor(n_estimators=20000)
#~ boosted_forest = RandomForestRegressor(n_estimators = 500, random_state = 42)
#~ boosted_svr = SVR()
#classifier = boosted_svr#AdaBoostRegressor(boosted_svr, n_estimators = 1000, random_state = 42)

classifier = SVR(kernel='rbf')
classifier.fit(trainset, trainlabels)
sum_error = 0
correct_classification = 0
for i, pred in enumerate(classifier.predict(testset)):
    #~ pred = 2.5
    #~ print "Prediction is:", pred
    #~ raw_input()
    sum_error += abs(pred - testlabels[i])
    if abs(pred-testlabels[i]) < (stddevs[i]*STDDEV_THRESHOLD):
        correct_classification += 1

print sum_error/len(testlabels)
print str(float(correct_classification)/len(testlabels)*100)+"%"

#~ print "Params", classifier.coef_
#~ y_pos = np.arange(len(classifier.coef_[0]))
#~ div_size = [12, 10, 16, 4, 12, 4, 5, 4, 1]
#~ coeffs = []
#~ for i in range(len(div_size)):
    #~ coeffs.append(classifier.coef_[0][i]*div_size[i])
#~ plt.bar(y_pos,coeffs)
#~ plt.show()
