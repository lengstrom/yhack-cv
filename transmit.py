import numpy as np
import cv2, pdb, requests

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    _, enc = cv2.imencode('.png', frame)
    s = enc.tostring()
    r__ = requests.post("http://127.0.0.1:5000/", files = {'i':s})
    if r__.text != '_':
        x, y, w, h = r.text.split(',')
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
