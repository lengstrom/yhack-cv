import cv2, pdb, socket, sys
import numpy as np

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', 5000)
buffer_size = 32768

sock.bind(server_address)
sock.listen(1)

current_stack = []
while True:
    # wait for connection
    connection, client_address = sock.accept()
    data = connection.recv(buffer_size)
    pdb.set_trace()










