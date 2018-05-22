import dlib
import openface
import cv2

img = cv2.imread("SCUT-FBP-1.jpg")
align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
bb = align.getLargestFaceBoundingBox(img)
landmarks = align.findLandmarks(img,bb)
