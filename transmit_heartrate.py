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
        x, y, w, h, ex, ey, ew, eh = map(lambda x: int(x), r__.text.split(','))
        ex += x
        ey += y
        cv2.rectangle(frame,(0,10),(10, 20),(0,255,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        if ew != 0:
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)
            fx, fy, fw, fh = find_forehead_with_eyes(x, y, w, h, ex, ey, ew, eh)
            cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(255,0,0),2)
        else:
            fx, fy, fw, fh = find_forehead_without_eyes(x, y, w, h)
            cv2.rectangle(frame,(fx,fy),(fx+fw,fy+fh),(255,0,0),2)
        
    start = time.time()
    cv2.imshow('frame', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    avg += time.time() - start
    #print avg/n
cap.release()
cv2.destroyAllWindows()
