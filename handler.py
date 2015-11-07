import pdb
import cv2
import tornado.httpserver
import tornado.ioloop
import numpy as np
import tornado.web
from tornado.options import define, options
tries = 0
last_face = (0, 0, 0, 0)
last_frame = 0

def convert_to_cv2_img(body):
    file_bytes = np.asarray(bytearray(body), dtype=np.uint8)
    return cv2.imdecode(file_bytes, cv2.CV_LOAD_IMAGE_UNCHANGED)

def find_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2, minSize=(250,250))
    return faces # each face is (x, y, w, h)

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
