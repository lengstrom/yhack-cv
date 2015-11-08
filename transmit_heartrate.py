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

def find_forehead_with_eyes(x, y, w, h, ex, ey, eh, ew):
#    fx =
    fw = w*0.40
    fx = x + w*(1 - 0.40)/2.
    eye_top = ey
    head_top = y
    dy = ey - y
    fh = 0.40635036496350365 * dy
    fy = y + (dy - fh)/2.4
    return map(lambda x: int(x), (fx, fy, fw, fh))

def find_forehead_without_eyes(x, y, w, h):
    fw = 0.39215686274509803 * w
    fx = x + (w - fw)/2
    fh = 0.06262230919765166 * h
    fy = y + h * 0.11741682974559686
    return map(lambda x: int(x), (fx, fy, fw, fh))

while True:
    go, frame = cap.read()
    frame = frame[40:680, 320:960]
    _, enc = cv2.imencode('.png', frame)
    s = enc.tostring()
    r__ = requests.post("http://127.0.0.1:5000/", data=s, headers=headers)#files = {'i':s})
    if r__.text != '_,_':
        if r__.text[-1] == '_':
            x, y, w, h = map(lambda x: int(x), r__.text.split(',')[:-1])
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        else:
            x, y, w, h, bpm, alpha, sec = r__.text.split(',')
            x, y, w, h, sec = map(lambda x: int(x), (x, y, w, h, sec))
            alpha = float(alpha)
            bpm = float(bpm)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            print bpm, alpha, sec

    start = time.time()
    cv2.imshow('frame', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    avg += time.time() - start

cap.release()
cv2.destroyAllWindows()
