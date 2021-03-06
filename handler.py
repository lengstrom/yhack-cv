import pdb, time, cv2, os, platform
import tornado.httpserver
import tornado.ioloop
import numpy as np
import tornado.web
from tornado.options import define, options

def convert_to_cv2_img(body): 
    file_bytes = np.asarray(bytearray(body), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_UNCHANGED)

def dist_between_faces(f1, f2):
    return (f1[0] + f1[2] - f2[0] - f2[2])**2 + (f1[1] + f1[3]- f2[1] - f2[3])**2

def ret_no_face(prev_face, tries):
    tries[0] += 1
    if tries[0] > MAX_NUM_TRIES:
        prev_face[0] = [0,0,0,0]
        return ()
    if prev_face[0][3] == 0:
        return ()
    else:
        return prev_face[0]

def serialize_face_pos(tup):
    a = ','.join(map(lambda x: str(x), tup))
    return a if a != '' else '_'

def ret_candidate(c, prev_face, tries):
    tries[0] = 0
    arr1 = prev_face[0]
    prev_face[0] = c
    return c

def find_forehead_with_eyes(x, y, w, h, ex, ey, eh, ew):
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

def get_forehead(prev_face):
    if prev_face[1][2] == 0:
        a = prev_face[0]
        return find_forehead_without_eyes(*a)
    else:
        a = prev_face[0] + prev_face[1]
        return find_forehead_with_eyes(*a)

def get_current_faces(candidates, prev_face, tries):
    is_prev = prev_face[0][2] != 0
    num_candidates = len(candidates)
    if num_candidates == 1: # only one face
        candidate = candidates[0]
        return ret_candidate(candidate, prev_face, tries)
    if num_candidates == 0:
        return ret_no_face(prev_face, tries)
    else:
        if not is_prev:
            return ret_no_face(prev_face, tries)
        else:
            min_ = ((), 999999)
            for i in candidates:
                dist = dist_between_faces(i, prev_face[0])
                if min_[1] > dist:
                    min_ = (i, dist)
            return ret_candidate(i, prev_face, tries)

def find_faces(img, prev_face, tries):
    #start = time.time()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.25, minNeighbors=3, minSize=(160,160))
    get_current_faces(faces, prev_face, tries)
    # with prev_face.get_lock():
    #     face = prev_face[0]
    #     w = face[2]
    #     if w > 0:
    #         x = face[0]
    #         y = face[1]
    #         h = face[3]
    #         eye_gray = gray[y:y+h, x:x+w]
    #         eyes = eye_cascade.detectMultiScale(eye_gray)
    #         n = 0
    #         avg_eye = [0, 0, 0, 0]
    #         for (ex,ey,ew,eh) in eyes:
    #             avg_eye[0] += ex
    #             avg_eye[1] += ey
    #             avg_eye[2] += ew
    #             avg_eye[3] += eh
    #             n += 1
    #         if n > 0:
    #             prev_face[1] = (ctypes.c_int * 4)(*(map(lambda x: x / n, avg_eye)))
    #         else:
    #             prev_face[1] = (ctypes.c_int * 4)()
    #     else:
    #         prev_face[1] = (ctypes.c_int * 4)()

class ImageHandler(tornado.web.RequestHandler):
    def initialize(self, prev_face, tries, forehead):
        self.prev_face = prev_face # face, forehead
        self.tries = tries
        self.forehead = forehead

    def get(self):
        self.render("index.html")

    def post(self):
        # if there's a face (or was in the last two seconds): send the coordinates of the image (x, y, h, w)
        # otherwise send '_
        img = convert_to_cv2_img(self.request.body) # return previous face coordinates
        find_faces(img, self.prev_face, self.tries)

        if prev_face[0][2] != 0: # if we have a prev face
            serialized = serialize_face_pos(prev_face[0])
            response_loc = serialized
        else: # if we don't have a previous face
            response_loc = '_' # return that we don't have a face
            
        self.write(response_loc)

if __name__ == "__main__":
    haar_path = '/usr/local/Cellar/opencv/2.4.12/share/OpenCV/haarcascades/' if platform.system() == 'Darwin' else r"C:\Users\Misha\Downloads\opencv\build\share\OpenCV\haarcascades" + chr(92)
    face_cascade = cv2.CascadeClassifier(haar_path + 'haarcascade_frontalface_default.xml')
    MAX_NUM_TRIES = 15
    MAX_DIST_BETWEEN_FACES = 2000
    mt_serialized = '_'
    #prev_face = (0, 0, 0, 0)
    mt_prev_face = [[0, 0, 0, 0], [0, 0, 0, 0]] #face, forehead
    tries = [0]
    forehead = []
    prev_face =  [[0,0,0,0],[0,0,0,0]]
    app = tornado.web.Application([
        (r"/", ImageHandler, dict(prev_face=prev_face, tries=tries, forehead=forehead))
    ])
    server = tornado.httpserver.HTTPServer(app)
    server.listen(5000)
    tornado.ioloop.IOLoop.instance().start()
