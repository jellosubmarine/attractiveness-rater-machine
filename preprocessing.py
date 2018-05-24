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

align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
DISPLAY_IMAGE = True # Marks whether the aim is to analyze or to visually inspect (essentially debug mode)
FOOL_AROUND = False # For testing our own images

# Read in image data
def process_image(path):
    global align

    img = cv2.imread(path)
    img_n,img_m,_ = img.shape
    
    midline = [8, 27, 28, 29, 51, 57, 62, 66] # All midline points except for two points on the tip of the nose, which are left out
    
    # Rotate until we are within 0.5 dgerees of straight
    for i in range(5):
        bb = align.getLargestFaceBoundingBox(img)
        landmarks = align.findLandmarks(img,bb)
        
        # Find line of symmetry and rotate image accordingly
        invslope, invintercept, _, _, _ = orgr.orthoregress([ landmarks[i][1] for i in midline ], [ landmarks[i][0] for i in midline ])
        # Rotate
        if abs(invslope) < 0.0087: # Approximately 0.5 degrees
            # everything is great, no rotation required
            break
        else:
            img = imutils.rotate_bound(img, atan(invslope)*180/pi)
    
    bb = align.getLargestFaceBoundingBox(img)
    landmarks = align.findLandmarks(img,bb)
    #~ invslope, invintercept, _, _, _ = scipy.stats.linregress([ landmarks[i][1] for i in midline ], [ landmarks[i][0] for i in midline ])
    invslope, invintercept, _, _, _ = orgr.orthoregress([ landmarks[i][1] for i in midline ], [ landmarks[i][0] for i in midline ])
    assert (abs(atan(invslope)*180/pi) < 0.5)
    
    # Crop the face part
    ys = [ p[1] for p in landmarks ]
    xs = [ p[0] for p in landmarks ]
    miny = min(ys)
    maxy = max(ys)
    minx = min(xs)
    maxx = max(xs)
    # Make sure axis of symmetry is in the middle of the image
    minx = int(floor(min(minx, 2*invintercept-maxx)))
    maxx = int(ceil(max(maxx, 2*invintercept-minx)))
    # Do the actual cropping
    img = img[miny:maxy, minx:maxx]
    # Scale image to width 500
    scale_factor = 500./(maxx-minx)
    img = cv2.resize(img, None, fx = scale_factor, fy = scale_factor, interpolation = cv2.INTER_CUBIC)
    
    # Re-adjust landmarks
    for i in range(len(landmarks)):
        x,y = landmarks[i]
        landmarks[i] = int(round((x-minx)*scale_factor)), int(round((y-miny)*scale_factor))
    
    if DISPLAY_IMAGE:
        # Mark face landmarks on image
        for i,point in enumerate(landmarks):
            #~ if i not in midline:
                #~ continue
            cv2.circle(img,point,2,(255,0,0),-1)
        
        # Draw line of symmetry on the face
        img_n,img_m,_ = img.shape
        cv2.line(img,(img_m/2,0),(img_m/2,img_n),(255,0,0),1)
    
    return landmarks, img

def main(args):
    img = None
    failed_images = []
    
    if not DISPLAY_IMAGE and not FOOL_AROUND:
        landmark_file = open("landmarks.txt", 'w')
    if not FOOL_AROUND:
        for i in range(1,501):
            try:
                print i
                
                landmarks, img = process_image("Data_Collection/SCUT-FBP-"+str(i)+".jpg")
                
                if not DISPLAY_IMAGE:
                    landmark_file.write(" ".join([ str(p[0]) + ' ' + str(p[1]) for p in landmarks ])+'\n')
                    cv2.imwrite("Adjusted_Data/SCUT-FBP-adjusted-"+str(i)+".jpg",img)
                else:
                    # Display image
                    cv2.imshow("Soust",img)
                    
                    while 1:
                        key = cv2.waitKey(1)
                        if key == 27:
                            break
                
            except Exception,e:
                print str(i) + " failed xd"
                failed_images.append(i)
    
    if not DISPLAY_IMAGE and not FOOL_AROUND:
        landmark_file.close()
    
    if FOOL_AROUND:
        test_landmark_file = open("testlandmarks.txt", 'a')
        for i in range(6,7):
            landmarks, img = process_image("Test_Data/Test-"+str(i)+".jpg")
            test_landmark_file.write(" ".join([ str(p[0]) + ' ' + str(p[1]) for p in landmarks ])+'\n')
            cv2.imwrite("Test_Data/Adjusted-"+str(i)+".jpg",img)
        test_landmark_file.close()
    
    print "Failed image numbers:",
    for i in failed_images:
        print i,
    print

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
