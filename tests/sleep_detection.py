# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
# import playsound
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
	return ear


# Passing '0` on windows, `-1` on linux
cap = cv2.VideoCapture(-1)

# HOG detector 
detector = dlib.get_frontal_face_detector()

# Pretrained face detector based on Historam of Oriented Gradients + Linear SVM
# datFile = './shape_predictor_68_face_landmarks.dat'
datFile = "./shape_predictor_68_face_landmarks (1).dat"
# dlib_facelandmark = dlib.shape_predictor(datFile)
predictor = dlib.shape_predictor(datFile)



# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# loop over frames from the video stream

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48
# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0 # eye close counter

# Eye open 
EO_COUNTER = 0 # eye open counter 
EO_CONSEC_FRAMES = 48 # approx 2s in real-time

while True:  
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:

        # Draw key chin keypoints   
        # face_landmarks = dlib_facelandmark(gray, face)
        face_landmarks = predictor(gray, face)
        
        shape = face_utils.shape_to_np(face_landmarks)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        lEAR = eye_aspect_ratio(leftEye)
        rEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (lEAR + rEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            EO_COUNTER = 0 
            COUNTER += 1
            # if the eyes were closed for a sufficient number of
            # then sound the alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                print('[INFO] Eye close for too long....')
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # otherwise, the eye aspect ratio is not below the blink
        # threshold, so reset the counter and alarm
        else:
            COUNTER = 0
            EO_COUNTER += 1 
            if EO_COUNTER >= EO_CONSEC_FRAMES: 
                print('[INFO] Eye OPEN for too long, please blink....')
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow("Face Landmarks", frame)
    key = cv2.waitKey(1)
    # Press 'Q' to exit
    if key == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()

