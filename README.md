# landmarkDetector
An OpenCV/imutils implementation to visualize and extract the facial action units of a detectable face.

This software requires cmake, as well as opencv-python, dlib, and imutils. These three packages will automatically install into your environment upon initial run.

This software will attempt to land 67 facial action units upon a detectable face and corroborate the real-time X-Y coordinates of each point into a numPy array.

landmarkDetector makes use of the OpenFace Landmark Extraction shape predictor, and is included in this repo.
Inspired by the article by Adrian Rosebrock: https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/ 
