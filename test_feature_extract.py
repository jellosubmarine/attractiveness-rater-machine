#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import openface
import cv2

img = cv2.imread("SCUT-FBP-1.jpg")
align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
bb = align.getLargestFaceBoundingBox(img)
landmarks = align.findLandmarks(img,bb)

cv2.rectangle(img,(bb.left(),bb.bottom()),(bb.right(),bb.top()),(0,255,0),3)

for point in landmarks:
    cv2.circle(img,point,3,(0,0,255),-1)

cv2.imshow("Soust",img)

while 1:
    key = cv2.waitKey(1)
    if key == 27:
        break
