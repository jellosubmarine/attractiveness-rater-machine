#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import openface
import cv2
import numpy as np
import scipy
from numpy.linalg import eig, inv
from math import atan, pi, floor, ceil
import imutils
import orthoregress as orgr
from preprocessing import process_image
import feature_creator
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd

def rate_image(img_path, classifier):
    landmarks, img = process_image(img_path)
    fv = feature_creator.create_feature_vec(landmarks)
    #Hotfix for a bug
    fv.insert(0,0)
    rating = classifier.predict([fv])[0]
    return img, rating

def train_classifier():
    features = pd.read_csv('featurevectors.csv')
    ratings = np.array(features['True_rating'])
    stddevs = np.array(features['Stdev'])
    features= features.drop('True_rating', axis = 1)
    features= features.drop('Stdev', axis = 1)
    classifier = AdaBoostRegressor(SVR(kernel='rbf'))
    classifier.fit(features, ratings)
    return classifier

def main(args):
    classifier = train_classifier()
    for i in range(4,5):
        img, rating = rate_image('Custom_images/'+str(i) +'.jpg', classifier)
        print str(i) + ' ' + str(round(rating, 2))
        cv2.imshow("Soust",img) 
        while 1:
            key = cv2.waitKey(1)
            if key == 27:
                break
    
if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))