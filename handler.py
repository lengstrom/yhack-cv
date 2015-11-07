import pdb
import cv2
import tornado.httpserver
import tornado.ioloop
import numpy as np
import tornado.web
from tornado.options import define, options

MAX_NUM_TRIES = 4
MAX_DIST_BETWEEN_FACES = 2000
tries = 0
prev_face = (0, 0, 0, 0)

def convert_to_cv2_img(body):
    file_bytes = np.asarray(bytearray(body), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_UNCHANGED)

def dist_between_faces(f1, f2):
    return (f1[0] + f1[2] - f2[0] - f2[2])**2 + (f1[1] + f1[3]- f2[1] - f2[3])**2

def ret_no_face():
    tries += 1
    if tries > MAX_NUM_TRIES:
        prev_face = (0, 0, 0, 0)
        return ()
    if prev_face[3] == 0:
        return ()
    else:
        return prev_face

def serialize_face_pos(tup):
    a = ','.join(map(lambda x: str(x), a))
    return a if a != '' else '_'

def ret_candidate(c):
    tries = 0
    prev_face = c
    return c

def get_current_faces(prev_face, candidates):
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
        ret_no_face()
    else:
        if not is_prev:
            return ret_no_face()
        else:
            min_ = (None, 999999)
            for i in candidates:
                dist = dist_between_faces(i, prev_face)
                if min_[1] > dist:
                    min_ = (i, dist)
            return ret_candidate(i)

def find_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(250,250))
    curr_face_pos = get_current_faces(faces)
    serialized = serialize_face(curr_face_pos)
    return serialized # each face is (x, y, w, h)

class ImageHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

    def post(self):
        # if there's a face (or was in the last two seconds): send the coordinates of the image (x, y, h, w)
        # otherwise send '_ 
        img = convert_to_cv2_img(self.request.files['i'][0]['body'])
        
        self.write(response)

if __name__ == "__main__":
    app = tornado.web.Application([
        (r"/", ImageHandler)
    ])
    server = tornado.httpserver.HTTPServer(app)
    server.listen(5000)
    tornado.ioloop.IOLoop.instance().start()
