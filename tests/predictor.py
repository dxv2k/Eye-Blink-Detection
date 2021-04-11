import cv2 
import dlib



from imutils import face_utils

# Passing '0` on windows, `-1` on linux
cap = cv2.VideoCapture(0)

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

# print(lStart,lEnd)
# print('-----------------')
# print(rStart,rEnd)


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
        # left
        # print(shape)

    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release() 
cv2.destroyAllWindows()