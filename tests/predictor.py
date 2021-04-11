import cv2 
import dlib

# Passing '0` on windows, `-1` on linux
cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()


# datFile = './shape_predictor_68_face_landmarks.dat'
datFile = r"tests/shape_predictor_68_face_landmarks (1).dat"
dlib_facelandmark = dlib.shape_predictor(datFile)

while True:  
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)

        for n in range(0, 16):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 1)


    cv2.imshow("Face Landmarks", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    cap.release() 
    cv2.destroyAllWindows()