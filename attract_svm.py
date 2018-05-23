#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

STDDEV_THRESHOLD = 0.5

# def read_in_feature_vectors(path):
#     fin = open(path, 'r')
#     t = [ line.split(' ') for line in fin.read().split('\n') if line ]
#     fin.close()
#     fvs = []
#     ratings = []
#     stddevs = []
#     for line in t:
#         fvs.append([ float(x) for x in line[:-2] ])
#         ratings.append(float(line[-2]))
#         stddevs.append(float(line[-1]))
#     return fvs, ratings, stddevs

# def divide_dataset(vectorset, labels, part_size):
#     datalen = len(vectorset)
#     shuffled_range = list(range(datalen))
#     random.shuffle(shuffled_range)
#     trainset = []
#     trainlabels = []
#     testset = []
#     testlabels = []
#     for i in shuffled_range[:int(datalen*part_size)]:
#         trainset.append(copy.deepcopy(vectorset[i]))
#         trainlabels.append(labels[i])
#     for i in shuffled_range[int(datalen*part_size):]:
#         testset.append(copy.deepcopy(vectorset[i]))
#         testlabels.append(labels[i])
#     return trainset, trainlabels, testset, testlabels

def graph_feature(xs, ys):
    plt.plot(xs, ys, 'b.')
    plt.show()

def test_dataset(features, ratings, stddevs, repeats):
    error_results = []
    stddev_results = []
    
    testlabels_all = []
    predlabels_all = []
    for repeat_time in range(repeats):
        print "Testing "+str(repeat_time+1)+'/'+str(repeats)
        
        trainset, testset, trainlabels, testlabels = train_test_split(features, ratings, test_size = 0.2)#, random_state = repeat_time)
        testlabels_all += list(testlabels)

        #~ classifier = SVR()
        #~ classifier = RandomForestRegressor(n_estimators=20000)
        #~ boosted_forest = RandomForestRegressor(n_estimators = 500, random_state = 42)
        #~ boosted_svr = SVR()
        #classifier = boosted_svr#AdaBoostRegressor(boosted_svr, n_estimators = 1000, random_state = 42)

        classifier = SVR(kernel='rbf')
        classifier.fit(trainset, trainlabels)
        sum_error = 0
        correct_classification = 0
        predlabels = []
        for i, pred in enumerate(classifier.predict(testset)):
            predlabels.append(pred)
            #~ pred = 2.6
            #~ print "Prediction is:", pred
            #~ raw_input()
            sum_error += abs(pred - testlabels[i])
            if abs(pred-testlabels[i]) < (stddevs[i]*STDDEV_THRESHOLD):
                correct_classification += 1
        
        predlabels_all += predlabels
        
        error_results.append(float(sum_error)/len(testlabels))
        stddev_results.append(float(correct_classification)/len(testlabels))
    
    graph_feature(testlabels_all, predlabels_all)
    
    print "Average error:", sum(error_results)/len(error_results)
    print "How many are within", STDDEV_THRESHOLD, "standard deviations:", sum(stddev_results)/len(stddev_results)
    print "Standard deviation for avg error:", np.std(np.array(error_results))
    print "Standard deviation for std results:", np.std(np.array(stddev_results))

#feature_vecs, ratings, stddevs = read_in_feature_vectors("featurevectors.csv")
features = pd.read_csv('featurevectors.csv')
ratings = np.array(features['True_rating'])
stddevs = np.array(features['Stdev'])
features= features.drop('True_rating', axis = 1)
features= features.drop('Stdev', axis = 1)

test_dataset(features, ratings, stddevs, 1000)

#~ print "Params", classifier.coef_
#~ y_pos = np.arange(len(classifier.coef_[0]))
#~ div_size = [12, 10, 16, 4, 12, 4, 5, 4, 1]
#~ coeffs = []
#~ for i in range(len(div_size)):
    #~ coeffs.append(classifier.coef_[0][i]*div_size[i])
#~ plt.bar(y_pos,coeffs)
#~ plt.show()
