#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dlib
import openface
import cv2
import numpy as np
from numpy.linalg import eig, inv
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

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else:
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

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
    center = ellipse_center(a)
    phi = ellipse_angle_of_rotation2(a)
    a, b = axes
    #~ arc = 4
    #~ R = np.arange(0,arc*np.pi, 0.01)
    #~ xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
    #~ yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
    #~ for i in range(len(xx)):
        #~ cv2.circle(img,(int(xx[i]),int(yy[i])),1,(0,0,255),-1)
    
    #~ cv2.imshow("Soust1",img)
    
    #~ while 1:
        #~ key = cv2.waitKey(1)
        #~ if key == 27:
            #~ break
    
    return float(b)/a, err

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
    ratio, err = face_ellipse(landmarks)
    fv.append(abs(ratio-((1 + 5 ** 0.5) / 2.0)))
    fv.append(err)
    return fv

# Dump feature vector data into a file
def write_features_to_file(path,feature_vecs):
    fout = open(path, 'w')
    for vec in feature_vecs:
        fout.write(" ".join([ str(x) for x in vec ]))
        fout.write('\n')
    fout.close()

def main(args):
    img = None
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
