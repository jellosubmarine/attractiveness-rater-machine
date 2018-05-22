#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import openface
import cv2
import numpy as np

img = cv2.imread("SCUT-FBP-42.jpg")
align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
bb = align.getLargestFaceBoundingBox(img)
landmarks = align.findLandmarks(img,bb)

cv2.rectangle(img,(bb.left(),bb.bottom()),(bb.right(),bb.top()),(0,255,0),2)
cv2.line(img,((bb.left()+bb.right())/2,bb.bottom()),((bb.left()+bb.right())/2,bb.top()),(0,255,0),2)
rekt = cv2.minAreaRect(np.array([ (int(x), int(y)) for (x,y) in landmarks ]))
boks = cv2.boxPoints(rekt)
#~ print boks
boks = np.int0(boks)

cv2.drawContours(img, [boks], 0, (0,0,255), 2)
cv2.line(img,((boks[0][0]+boks[1][0])/2,(boks[0][1]+boks[1][1])/2),((boks[2][0]+boks[3][0])/2,(boks[2][1]+boks[3][1])/2),(0,0,255),2)
for point in landmarks:
    cv2.circle(img,point,2,(255,0,0),-1)

cv2.line(img,landmarks[8],landmarks[27],(255,0,0),2)

cv2.imshow("Soust",img)

while 1:
    key = cv2.waitKey(1)
    if key == 27:
        break
