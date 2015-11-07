import pdb, time, cv2, ctypes
import tornado.httpserver
import tornado.ioloop
import numpy as np
import tornado.web
from tornado.options import define, options
import multiprocessing as mp

face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/2.4.12/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
MAX_NUM_TRIES = 4
MAX_DIST_BETWEEN_FACES = 2000
tries = 0
mt_serialized = '_'
#prev_face = (0, 0, 0, 0)
mt_prev_face = [[0, 0, 0, 0], [0, 0, 0, 0]] #face, forehead
proc = mp.Process()
proc.start()

def convert_to_cv2_img(body): 
    file_bytes = np.asarray(bytearray(body), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_UNCHANGED)

def dist_between_faces(f1, f2):
    return (f1[0] + f1[2] - f2[0] - f2[2])**2 + (f1[1] + f1[3]- f2[1] - f2[3])**2

def ret_no_face(prev_face):
    global tries
    tries += 1
    if tries > MAX_NUM_TRIES:
        with prev_face.get_lock():
            prev_face[0] = (ctypes.c_int*4)()
        return ()
    if prev_face[0][3] == 0:
        return ()
    else:
        return prev_face[0]

def serialize_face_pos(tup):
    a = ','.join(map(lambda x: str(x), tup))
    return a if a != '' else '_'

def ret_candidate(c, prev_face):
    global tries
    tries = 0
    arr1 = prev_face[0]
    with prev_face.get_lock():
        prev_face[0] = (ctypes.c_int * 4)(*c)
    return c

def dispatch_proc(img, prev_face):
    global proc
    proc = mp.Process(target=find_faces, args=(img,prev_face))
    # proc = Process(target=sleep10)
    proc.start()

def get_current_faces(candidates, prev_face):
    is_prev = prev_face[0][2] != 0
    num_candidates = len(candidates)
    if num_candidates == 1: # only one face
        candidate = candidates[0]
        # dist = dist_between_faces(prev_face[0], candidate)
        # if dist > 2000:
        #     return ret_no_face(prev_face)
        # else:
        return ret_candidate(candidate, prev_face)
    if num_candidates == 0:
        return ret_no_face(prev_face)
    else:
        if not is_prev:
            return ret_no_face(prev_face)
        else:
            min_ = ((), 999999)
            for i in candidates:
                dist = dist_between_faces(i, prev_face[0])
                if min_[1] > dist:
                    min_ = (i, dist)
            return ret_candidate(i, prev_face)

def find_faces(img, prev_face):
    #start = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(250,250))
    curr_face_pos = get_current_faces(faces, prev_face)
    # don't return anything bc this is in another thread
    # eventually implement forehead finding method here

class ImageHandler(tornado.web.RequestHandler):
    def initialize(self, prev_face):
        self.prev_face = prev_face # face, forehead

    def get(self):
        self.render("index.html")

    def post(self):
        # if there's a face (or was in the last two seconds): send the coordinates of the image (x, y, h, w)
        # otherwise send '_
        global proc
        global mt_prev_face
        global mt_serialized
        img = convert_to_cv2_img(self.request.body) # return previous face coordinates
        if not proc.is_alive(): # proc finished
            for i in range(2):
                curr_mt = mt_prev_face[i]
                curr_pf = self.prev_face[i]
                for j in range(4):
                    curr_mt[j] = int(curr_pf[j])

            mt_serialized = serialize_face_pos(mt_prev_face[0])
            dispatch_proc(img, self.prev_face) # dispatch new process
        else:
            pass

        if mt_prev_face[0][2] != 0: # if we have a prev face
            response_loc = mt_serialized
        else: # if we don't have a previous face
            response_loc = '_' # return that we don't have a face

        self.write(response_loc)

if __name__ == "__main__":
    fin = ((ctypes.c_int*4) * 2)()
    prev_face =  mp.Array(type(fin[0]), fin)
    app = tornado.web.Application([
        (r"/", ImageHandler, dict(prev_face=prev_face))
    ])
    server = tornado.httpserver.HTTPServer(app)
    server.listen(5000)
    tornado.ioloop.IOLoop.instance().start()
