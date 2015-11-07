import numpy as np
import pdb
import cv2

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
n = 0
while(True):
    n+=1
    # Capture frame-by-frame
    ret, frame = cap.read()
    img = frame[10:710, 250:1030]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(220,250))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Display the resulting frame
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print n
cap.release()
cv2.destroyAllWindows()
