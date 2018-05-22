#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import openface
import cv2
import numpy as np

# Read in image data
img = cv2.imread("SCUT-FBP-1.jpg")
img_n,img_m,_ = img.shape
align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
img = align.align(500,img)
bb = align.getLargestFaceBoundingBox(img)
landmarks = align.findLandmarks(img,bb)

#~ # Get some versions of bounding boxes for the face
#~ cv2.rectangle(img,(bb.left(),bb.bottom()),(bb.right(),bb.top()),(0,255,0),2)
#~ cv2.line(img,((bb.left()+bb.right())/2,bb.bottom()),((bb.left()+bb.right())/2,bb.top()),(0,255,0),2)
#~ rekt = cv2.minAreaRect(np.array([ (int(x), int(y)) for (x,y) in landmarks ]))
#~ boks = cv2.boxPoints(rekt)
#~ boks = np.int0(boks)
#~ cv2.drawContours(img, [boks], 0, (0,0,255), 2)
#~ cv2.line(img,((boks[0][0]+boks[1][0])/2,(boks[0][1]+boks[1][1])/2),((boks[2][0]+boks[3][0])/2,(boks[2][1]+boks[3][1])/2),(0,0,255),2)

# Mark face landmarks on image
for point in landmarks:
    cv2.circle(img,point,2,(255,0,0),-1)

# Fit linear function to face's line of symmetry
symmetry_points = [8, 27, 28, 29, 30, 51, 57, 62, 66]
xs = [ landmarks[i][0] for i in symmetry_points ]
ys = [ landmarks[i][1] for i in symmetry_points ]
fit = np.polyfit(ys,xs,1) # note: flipped
fit_fn = np.poly1d(fit)
cv2.line(img,(int(fit_fn(0)),0),(int(fit_fn(img_n)),img_n),(255,0,0),2)

# Display image
cv2.imshow("Soust",img)

while 1:
    key = cv2.waitKey(1)
    if key == 27:
        break
