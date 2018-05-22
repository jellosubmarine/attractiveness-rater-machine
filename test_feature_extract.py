#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import openface
import cv2
import numpy as np

import sym_mappings as sym

align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")

# Read in image data
def process_image(path):
    global align

    img = cv2.imread(path)
    img = align.align(501,img)
    img_n,img_m,_ = img.shape
    bb = align.getLargestFaceBoundingBox(img)
    landmarks = align.findLandmarks(img,bb)
    
    # Mark face landmarks on image
    for point in landmarks:
        cv2.circle(img,point,2,(255,0,0),-1)
    
    # Create line of symmetry
    cv2.line(img,(img_m/2,img_n),(img_m/2,0),(255,0,0),2)
    
    return landmarks, img

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

# Create feature vector for image
def create_feature_vec(landmarks):
    fv = []
    fv.append(eval_symmetry(landmarks, sym.eye_sym))
    fv.append(eval_symmetry(landmarks, sym.brow_sym))
    fv.append(eval_symmetry(landmarks, sym.face_sym))
    fv.append(eval_symmetry(landmarks, sym.nose_sym))
    fv.append(eval_symmetry(landmarks, sym.outer_mouth_sym))
    fv.append(eval_symmetry(landmarks, sym.inner_mouth_sym))
    fv.append(eval_center(landmarks, sym.nose_center))
    fv.append(eval_center(landmarks, sym.mouth_center)) 
    fv.append(eval_center(landmarks, sym.chin_center))
    fv.append(eval_nose_length(landmarks))
    fv.append(eval_nose_roundness(landmarks))
    return fv

# Dump feature vector data into a file
def write_features_to_file(path,feature_vecs):
    fout = open(path, 'w')
    for vec in feature_vecs:
        fout.write(" ".join([ str(x) for x in vec ]))
        fout.write('\n')
    fout.close()

def main(args):
    feature_vecs = []
    for i in range(1,501):
        try:
            print i
            landmarks, img = process_image("Data_Collection/SCUT-FBP-"+str(i)+".jpg")
            feature_vecs.append(create_feature_vec(landmarks))
        except Exception,e:
            print str(i) + " failed xd"

    write_features_to_file("featurevectors.txt",feature_vecs)

    # Display image
    #~ cv2.imshow("Soust",img)
    
    #~ while 1:
        #~ key = cv2.waitKey(1)
        #~ if key == 27:
            #~ break

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
