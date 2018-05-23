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
import pandas as pd

SHOW_PLOTS = True

#Evaluate nose length
def eval_nose_length(landmarks):
    return np.linalg.norm(np.asarray(landmarks[27])-np.asarray(landmarks[33]))

#Evaluate nose curvature
def eval_nose_roundness(landmarks):
    nose_points = landmarks[31:36]
    return np.polyfit(nose_points[:][0],nose_points[:][1], 2)[2]

# Evaluate (distance between pupils)/(width of face)
def eval_pupil_face_ratio(landmarks):
    pupil_dist = float(landmarks[45][0]+landmarks[42][0] - landmarks[39][0]-landmarks[36][0])/2
    face_width = (landmarks[16][0]-landmarks[0][0])
    return float(pupil_dist)/face_width

# Evaluate (nose to inside of eye)/(eye width)
def eval_nose_to_eye_by_eye_width(landmarks):
    nose_to_eye = float(landmarks[42][0]-landmarks[39][0])/2
    eye_width = float(landmarks[45][0]-landmarks[42][0]+landmarks[39][0]-landmarks[36][0])/2
    return nose_to_eye/eye_width

# Evaluate (face side to eye outside)/(eye outside to nose center)
def eval_face_side_to_eye_outside_by_eye_to_nose(landmarks):
    face_to_eye = float(landmarks[16][0]-landmarks[45][0]+landmarks[36][0]-landmarks[0][0])/2
    eye_to_nose = float(landmarks[45][0]-landmarks[36][0])/2
    return face_to_eye/eye_to_nose

# Evaluate horizontal distances in mouth
def eval_mouth_horizontal(landmarks):
    mouth_short = float(landmarks[54][0]-landmarks[52][0]+landmarks[50][0]-landmarks[48][0])/2
    mouth_long = float(landmarks[54][0]-landmarks[50][0]+landmarks[52][0]-landmarks[48][0])/2
    return mouth_short/mouth_long

# Evaluate face side to inner eyebrow to face other side
def eval_face_side_to_brow_inside_to_face_side(landmarks):
    face_to_brow = float(landmarks[16][0]-landmarks[22][0]+landmarks[21][0]-landmarks[0][0])/2
    brow_to_other = float(landmarks[16][0]-landmarks[21][0]+landmarks[22][0]-landmarks[0][0])/2
    return face_to_brow/brow_to_other

# Evaluate face side to inner eye to face other side
def eval_face_side_to_eye_inside_to_face_side(landmarks):
    face_to_eye = float(landmarks[16][0]-landmarks[42][0]+landmarks[39][0]-landmarks[0][0])/2
    eye_to_other = float(landmarks[16][0]-landmarks[39][0]+landmarks[42][0]-landmarks[0][0])/2
    return face_to_eye/eye_to_other

# Evaluate face side to nose side to face other side
def eval_face_side_to_nose_side_to_face_side(landmarks):
    face_to_nose = float(landmarks[16][0]-landmarks[35][0]+landmarks[31][0]-landmarks[0][0])/2
    nose_to_other = float(landmarks[16][0]-landmarks[31][0]+landmarks[35][0]-landmarks[0][0])/2
    return face_to_nose/nose_to_other

# Evaluate nose vertical proportions
def eval_nose_vertical_prop(landmarks):
    nose_bottom = float(landmarks[33][1]-landmarks[30][1])
    nose_top = landmarks[30][1]-landmarks[27][1]
    return nose_bottom/nose_top

# Evaluate mouth size
def eval_mouth_size(landmarks):
    return landmarks[54][0]-landmarks[48][0]

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
    # fv.append(eval_symmetry(landmarks, sym.eye_sym)) # r = 0.033
    # fv.append(eval_symmetry(landmarks, sym.brow_sym)) # r = 0.034
    # fv.append(eval_symmetry(landmarks, sym.face_sym)) # r = 0.060
    # fv.append(eval_symmetry(landmarks, sym.nose_sym)) # r = 0.010
    # fv.append(eval_symmetry(landmarks, sym.outer_mouth_sym)) # r = 0.125
    # fv.append(eval_symmetry(landmarks, sym.inner_mouth_sym)) # r = 0.109
    # fv.append(eval_center(landmarks, sym.nose_center)) # r = 0.090
    # fv.append(eval_center(landmarks, sym.mouth_center)) # r = 0.050
    # fv.append(eval_center(landmarks, sym.chin_center)) # r = 0.042
    fv.append(eval_nose_length(landmarks)) # r = 0.307
    # fv.append(eval_nose_roundness(landmarks)) # r = -0.057 (looks weird as a plot, might actually be better with a few outliers that mess things up)
    ratio, err = face_ellipse(landmarks) 
    #fv.append(abs(ratio-((1 + 5 ** 0.5) / 2.0))) # r = -0.372
    fv.append(ratio) # r = 0.630
    # fv.append(err) # r = 0.152
    #~ fv.append(eval_pupil_face_ratio(landmarks)) # r = 0.190
    #~ fv.append(eval_nose_to_eye_by_eye_width(landmarks)) # r = -0.097
    #~ fv.append(eval_face_side_to_eye_outside_by_eye_to_nose(landmarks)) # r = -0.223
    #~ fv.append(eval_mouth_horizontal(landmarks)) # r = -0.151
    fv.append(eval_face_side_to_brow_inside_to_face_side(landmarks)) # r = -0.280
    #~ fv.append(eval_face_side_to_eye_inside_to_face_side(landmarks)) # r = -0.106
    #~ fv.append(eval_face_side_to_nose_side_to_face_side(landmarks)) # r = 0.183
    fv.append(eval_nose_vertical_prop(landmarks)) # r = -0.435
    #~ fv.append(eval_mouth_size(landmarks)) # r = 0.021
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
    
def graph_feature(feature_name, feature, ratings):
    plt.plot(feature, ratings, 'b.')
    plt.title(feature_name+", r = "+str(pearsonr(feature, ratings)[0]))
    plt.show()

def main(args):
    img = None
    feature_vecs = []
    failed_images = []
    
    ratings, stddevs = read_in_ratings("rating.csv")
    
    all_landmarks = read_in_landmarks("landmarks.txt")
    #Add feature vector labels here!
    
    labels = ['Nose_length', 'Face_ellipse_ratio', 'Face_side_with_brow',
              'Nose_vertical_prop',
              'True_rating', 'Stdev']
    df = pd.DataFrame.from_records([], columns=labels)
    print df
    for i in range(1,501):
        try:
            #print i
            
            landmarks = all_landmarks[i-1]
            
            df = df.append(pd.DataFrame([tuple(create_feature_vec(landmarks)+[ratings[i-1],stddevs[i-1]])], columns=labels))
        except Exception,e:
            print str(i) + " failed xd"
            failed_images.append(i)
    
    if SHOW_PLOTS:
        for i in range(len(labels)-2):
            graph_feature(labels[i], df[labels[i]], ratings)
    
    #write_features_to_file("featurevectors.csv",feature_vecs)
    df.to_csv('featurevectors.csv')
    
    print "Feature vector creation completed"
    print "Failed image numbers:",
    for i in failed_images:
        print i,
    print

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
