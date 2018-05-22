#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from sklearn.svm import LinearSVC

def read_in_ratings(path):
    fin = open(path,'r')
    t = [ line.split(',') for line in fin.read().split('\n') if line ][1:]
    fin.close()
    ratings = [ float(line[1].strip()) for line in t ]
    return ratings

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
        trainlabels.append(labels[vectorset[i]])
    for i in shuffled_range[int(datalen*part_size):]:
        testset.append(copy.deepcopy(vectorset[i]))
        testlabels.append(labels[vectorset[i]])
    return trainset, trainlabels, testset, testlabels

ratings = read_in_ratings("rating.csv")

trainset, trainlabels, testset, testlabels = divide_dataset(feature_vecs, ratings, 0.9)

classifier = LinearSVC()
classifier.fit(trainica, trainlabels)
