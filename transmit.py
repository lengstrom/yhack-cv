import numpy as np
import cv2, pdb, requests, time

cap = cv2.VideoCapture(0)
n = 0.
avg = 0.

# print "preprocessing..."
# go = True
# frames = []
# while go:
#     go, frame = cap.read()
#     if go:
#         frame = frame[40:680, 320:960]
#         _, enc = cv2.imencode('.png', frame)
#         s = enc.tostring()
#         frames.append((s, frame))
headers = {'Content-Type': 'application/octet-stream'}

while True:
    go, frame = cap.read()
    frame = frame[40:680, 320:960]
    _, enc = cv2.imencode('.png', frame)
    s = enc.tostring()
    r__ = requests.post("http://127.0.0.1:5000/", data=s, headers=headers)#files = {'i':s})
    if r__.text != '_':
        x, y, w, h = map(lambda x: int(x), r__.text.split(','))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    start = time.time()
    cv2.imshow('frame', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    avg += time.time() - start
    #print avg/n
cap.release()
cv2.destroyAllWindows()
