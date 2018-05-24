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
from sklearn.model_selection import RandomizedSearchCV

STDDEV_THRESHOLD = 1.0

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
        
        classifier = AdaBoostRegressor(SVR(kernel='rbf'))

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

#Function to find best parameters for the algorithms
def find_best_parameters(features, ratings):
    param_dist = {
        'n_estimators': [50, 100, 150],
        'learning_rate' : [0.01,0.05,0.1,0.3,1,1.5,2],
        'loss' : ['linear', 'square', 'exponential']
        }
    pre_gs_inst = RandomizedSearchCV(AdaBoostRegressor(SVR(kernel='rbf')),
    param_distributions = param_dist,
    cv=3,
    n_iter = 10,
    n_jobs=-1)

    pre_gs_inst.fit(features, ratings)

    print pre_gs_inst.best_params_

    param_dist = {
        'epsilon': [0.05,0.07,0.1,0.13,0.15],
        'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
        'shrinking' : [True, False]
        }
    pre_gs_inst = RandomizedSearchCV(SVR(),
    param_distributions = param_dist,
    cv=3,
    n_iter = 10,
    n_jobs=-1)

    pre_gs_inst.fit(features, ratings)

    print pre_gs_inst.best_params_
#feature_vecs, ratings, stddevs = read_in_feature_vectors("featurevectors.csv")
features = pd.read_csv('featurevectors.csv')
ratings = np.array(features['True_rating'])
stddevs = np.array(features['Stdev'])
features= features.drop('True_rating', axis = 1)
features= features.drop('Stdev', axis = 1)

# test_dataset(features, ratings, stddevs, 100)

# FOR THE LULZ
testfeatures = pd.read_csv('testfeaturevectors.csv')
classifier = AdaBoostRegressor(SVR(kernel='rbf'))
classifier.fit(features, ratings)
print classifier.predict(np.array(testfeatures))

# find_best_parameters(features, ratings)

#~ print "Params", classifier.coef_
#~ y_pos = np.arange(len(classifier.coef_[0]))
#~ div_size = [12, 10, 16, 4, 12, 4, 5, 4, 1]
#~ coeffs = []
#~ for i in range(len(div_size)):
    #~ coeffs.append(classifier.coef_[0][i]*div_size[i])
#~ plt.bar(y_pos,coeffs)
#~ plt.show()
