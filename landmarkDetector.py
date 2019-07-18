"""
landmarkDetector, a facial landmark detection tool.

This software will attempt to land 67 facial action units upon a detectable face and corroborate the real-time X-Y coordinates of each point into a numPy array.
landmarkDetector makes use of the OpenFace Landmark Extraction shape predictor, and is included in this repo.

Inspired by the article by Adrian Rosebrock: https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/ 

Written by Kaleb Byrum, from Louisville with <3
"""

import os

os.system("pip3 install opencv-python")
os.system("pip3 install imutils")
os.system("pip3 install dlib")

#import necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

#Parse the input arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-r", "--picamera", type=int, default=-1, help = "Which camera should be used")
# args = vars(ap.parse_args())

#Shape Predictor should be the path to dlibs pre-trained facial landmark predictor...

#Initialize dlib's HOG-based face detector and load facial predictor
print("Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Initialize VideoStream
print("Initializing camera stream...")
vs = VideoStream().start()

#Loop over frames within video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    #Loop over face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape) #REAL-TIME NUMPY ARRAY OF DECTECTED COORDS
        #print(shape)

        #loop over x,y coords and draw on image
        dotCount = 0
        for (x, y) in shape: #add text to each circle
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(dotCount), (x, y), font, 0.5, (0,255,0), 2, cv2.LINE_AA)
            dotCount += 1
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #time.sleep(1)

    #Q will exit program
    if key == ord("q"):
        break

#Cleanup and destroy windows
cv2.destroyAllWindows
vs.stop()
