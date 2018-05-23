#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
from math import atan, pi, floor, ceil
from scipy.stats.stats import pearsonr
import sym_mappings as sym

#Evaluate nose length
def eval_nose_length(landmarks):
    return np.linalg.norm(np.asarray(landmarks[27])-np.asarray(landmarks[33]))

#Evaluate nose curvature
def eval_nose_roundness(landmarks):
    nose_points = landmarks[31:36]
    return np.polyfit(nose_points[:][0],nose_points[:][1], 2)[2]

# Create symmetry evaluations for facial features
def eval_symmetry(landmarks, sym_mapping):
    s = 0
    for i,j in sym_mapping:
        a = landmarks[i]
        b = landmarks[j]
        s += abs(abs(a[0]-250)-abs(b[0]-250))
        s += abs(a[1]-b[1])
    return s

# Evaluate whether features in the middle of the face are actually in the middle
def eval_center(landmarks, middle_list):
    return sum([ abs(landmarks[i][0]-250) for i in middle_list ])

# Evaluate roundness of face and the eccentricity of ellipse
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a,np.abs(E[n])

def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def face_ellipse(landmarks):
    # Get the points that determine the face
    face_points = landmarks[:17]
    # Fit an ellipse to the points
    a, err = fitEllipse(np.array([ p[0] for p in face_points ]),np.array([ p[1] for p in face_points ]))
    axes = ellipse_axis_length(a)
    a, b = axes
    return float(b)/a, err

# Create feature vector for image
def create_feature_vec(landmarks):
    fv = []
    fv.append(eval_symmetry(landmarks, sym.eye_sym)) # r = 0.033
    fv.append(eval_symmetry(landmarks, sym.brow_sym)) # r = 0.034
    fv.append(eval_symmetry(landmarks, sym.face_sym)) # r = 0.060
    fv.append(eval_symmetry(landmarks, sym.nose_sym)) # r = 0.010
    fv.append(eval_symmetry(landmarks, sym.outer_mouth_sym)) # r = 0.125
    fv.append(eval_symmetry(landmarks, sym.inner_mouth_sym)) # r = 0.109
    fv.append(eval_center(landmarks, sym.nose_center)) # r = 0.090
    fv.append(eval_center(landmarks, sym.mouth_center)) # r = 0.050
    fv.append(eval_center(landmarks, sym.chin_center)) # r = 0.042
    fv.append(eval_nose_length(landmarks)) # r = 0.307
    fv.append(eval_nose_roundness(landmarks)) # r = -0.057 (looks weird as a plot, might actually be better with a few outliers that mess things up)
    ratio, err = face_ellipse(landmarks) 
    fv.append(abs(ratio-((1 + 5 ** 0.5) / 2.0))) # r = -0.372
    fv.append(ratio) # r = 0.630
    fv.append(err) # r = 0.152
    return fv

# Uses just plain landmarks for feature vectors
def create_feature_vec_2(landmarks):
    fv = []
    for lm in landmarks:
        fv.append(lm[0])
        fv.append(lm[1])
    return fv

# Dump feature vector data into a file
def write_features_to_file(path,feature_vecs):
    fout = open(path, 'w')
    for vec in feature_vecs:
        fout.write(" ".join([ str(x) for x in vec ]))
        fout.write('\n')
    fout.close()

# Read in landmarks from landmark file
def read_in_landmarks(path):
    fin = open(path, 'r')
    t = [ line.split(' ') for line in fin.read().split('\n') if line ]
    fin.close()
    
    landmarks = []
    for line in t:
        lm = []
        assert (len(line)%2 == 0)
        for i in range(len(line)/2):
            lm.append((int(line[2*i]),int(line[2*i+1])))
        landmarks.append(lm)
    
    return landmarks

# Read in ratings and rating standard deviation info
def read_in_ratings(path):
    fin = open(path,'r')
    t = [ line.split(',') for line in fin.read().split('\n') if line ][1:]
    fin.close()
    ratings = [ float(line[1].strip()) for line in t ]
    stddevs = [ float(line[2].strip()) for line in t ]
    return ratings, stddevs
    
def graph_feature(feature_vecs, ratings, feature_index):
    xs = [ v[feature_index] for v in feature_vecs ]
    plt.plot(xs, ratings, 'b.')
    plt.title(str(feature_index)+", r = "+str(pearsonr(xs, ratings)[0]))
    plt.show()

def main(args):
    img = None
    feature_vecs = []
    failed_images = []
    
    ratings, stddevs = read_in_ratings("rating.csv")
    
    all_landmarks = read_in_landmarks("landmarks.txt")
    
    for i in range(1,501):
        try:
            print i
            
            landmarks = all_landmarks[i-1]
            
            feature_vecs.append(create_feature_vec(landmarks)+[ratings[i-1],stddevs[i-1]])
        except Exception,e:
            print str(i) + " failed xd"
            failed_images.append(i)
    
    for i in range(len(feature_vecs[0])-2):
        graph_feature(feature_vecs, ratings, i)
    
    write_features_to_file("featurevectors.txt",feature_vecs)
    print "Feature vector creation completed"
    print "Failed image numbers:",
    for i in failed_images:
        print i,
    print

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
