import pdb
import cv2
import tornado.httpserver
import tornado.ioloop
import numpy as np
import tornado.web
from tornado.options import define, options
from multiprocessing import Process

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
MAX_NUM_TRIES = 4
serialized = '_'
MAX_DIST_BETWEEN_FACES = 2000
tries = 0
mt_serialized = '_'
prev_face = (0, 0, 0, 0)
mt_prev_face = (0, 0, 0, 0)
proc = Process()

def convert_to_cv2_img(body):
    file_bytes = np.asarray(bytearray(body), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_UNCHANGED)

def dist_between_faces(f1, f2):
    return (f1[0] + f1[2] - f2[0] - f2[2])**2 + (f1[1] + f1[3]- f2[1] - f2[3])**2

def ret_no_face():
    global tries
    global prev_face
    tries += 1
    if tries > MAX_NUM_TRIES:
        prev_face = (0, 0, 0, 0)
        return ()
    if prev_face[3] == 0:
        return ()
    else:
        return prev_face

def serialize_face_pos(tup):
    a = ','.join(map(lambda x: str(x), tup))
    return a if a != '' else '_'

def ret_candidate(c):
    global tries
    global prev_face
    tries = 0
    prev_face = c
    return c

def dispatch_proc(img):
    print "dispatched proc."
    global proc
    proc = Process(target=find_faces, args=(img,))
    proc.start()

def get_current_faces(candidates):
    is_prev = prev_face[2] != 0
    num_candidates = len(candidates)
    if num_candidates == 1: # only one face
        candidate = candidates[0]
        if not is_prev:
            return ret_candidate(candidate)
        dist = dist_between_faces(prev_face, candidate)
        if dist_between_faces > 2000:
            return ret_no_face()
        else:
            return ret_candidate(candidate)
    if num_candidates == 0:
        return ret_no_face()
    else:
        if not is_prev:
            return ret_no_face()
        else:
            min_ = ((), 999999)
            for i in candidates:
                dist = dist_between_faces(i, prev_face)
                if min_[1] > dist:
                    min_ = (i, dist)
            return ret_candidate(i)

def find_faces(img):
    global serialized
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(250,250))
    curr_face_pos = get_current_faces(faces)
    serialized = serialize_face_pos(curr_face_pos)
    print "    proc finished"
    # don't return anything bc this is in another thread
    # eventually implement forehead finding method here

class ImageHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

    def post(self):
        # if there's a face (or was in the last two seconds): send the coordinates of the image (x, y, h, w)
        # otherwise send '_
        print "    posted..."
        global proc
        global mt_prev_face
        global serialized
        global mt_serialized
        global prev_face
        img = convert_to_cv2_img(self.request.files['i'][0]['body']) # return previous face coordinates
        if not proc.is_alive(): # proc finished
            mt_prev_face = prev_face[:] # reassign prev_face area
            mt_serialized = serialized[:]
            dispatch_proc(img) # dispatch new process

        if mt_prev_face[2] != 0: # if we have a prev face
            response_loc = mt_serialized
        else: # if we don't have a previous face
            response_loc = serialize_face_pos(()) # return that we don't have a face

        print mt_serialized
        self.write(response_loc)

if __name__ == "__main__":
    app = tornado.web.Application([
        (r"/", ImageHandler)
    ])
    server = tornado.httpserver.HTTPServer(app)
    server.listen(5000)
    tornado.ioloop.IOLoop.instance().start()
