import cv2 
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist

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
cap = cv2.VideoCapture(0)

# HOG detector 
hog_face_detector = dlib.get_frontal_face_detector()

# Pretrained face detector based on Historam of Oriented Gradients + Linear SVM
# datFile = './shape_predictor_68_face_landmarks.dat'
datFile = r"tests/shape_predictor_68_face_landmarks (1).dat"
# dlib_facelandmark = dlib.shape_predictor(datFile)
predictor = dlib.shape_predictor(datFile)


# grab the indexes of the facial landmarks for the left and
# right eye, respectively
# Ref: https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #42,48 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] # 36, 42

EYE_THRESHOLD = 0.25

l_val = [] 
r_val = [] 

while True:  
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        # Draw key chin keypoints   
        # face_landmarks = dlib_facelandmark(gray, face)
        face_landmarks = predictor(gray, face)
        for n in range(0, 16):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)

        # TODO: detect eye and draw bb
        # convert face_landmarks to np 
        shape = face_utils.shape_to_np(face_landmarks)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        lEAR = eye_aspect_ratio(leftEye)
        rEAR = eye_aspect_ratio(rightEye)
        if (lEAR <= EYE_THRESHOLD) and (rEAR <= EYE_THRESHOLD): 
            print('Both eye close')

        l_val.append(lEAR)
        r_val.append(rEAR)


    cv2.imshow("Face Landmarks", frame)
    key = cv2.waitKey(1)
    # Press 'Q' to exit
    if key == ord('q'):
        break

cap.release() 
cv2.destroyAllWindows()


# Visualize both eye threshold 
import matplotlib.pyplot as plt 
import pandas as pd

def visualize(val1,val2, label1,label2): 
    # df1 = pd.DataFrame(val1)
    # df2 = pd.DataFrame(val2)
    plt.plot(val1,label=label1)
    plt.plot(val2,label=label2)
    plt.show()


visualize(l_val,r_val,label1='left',label2='right')
