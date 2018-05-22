#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import openface
import cv2
import numpy as np

# Read in image data
img = cv2.imread("SCUT-FBP-42.jpg")
align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
img = align.align(501,img)
img_n,img_m,_ = img.shape
bb = align.getLargestFaceBoundingBox(img)
landmarks = align.findLandmarks(img,bb)

# Mark face landmarks on image
for point in landmarks:
    cv2.circle(img,point,2,(255,0,0),-1)

# Fit linear function to face's line of symmetry
cv2.line(img,(img_m/2,img_n),(img_m/2,0),(255,0,0),2)

# Create symmetry evaluations for facial features
def eval_symmetry(landmarks, sym_mapping):
    s = 0
    for i,j in sym_mapping:
        a = landmarks[i]
        b = landmarks[j]
        s += (abs(a[0]-b[0]) + abs(a[1]-b[1]))
    return s

# Evaluate whether features in the middle of the face are actually in the middle
def eval_center(landmarks, middle_list):
    return sum([ abs(landmarks[i][0]-250) for i in middle_list ])

# Display image
cv2.imshow("Soust",img)

while 1:
    key = cv2.waitKey(1)
    if key == 27:
        break
