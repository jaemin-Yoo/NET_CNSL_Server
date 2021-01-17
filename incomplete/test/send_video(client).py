#Client
import socket
import cv2
import numpy
from queue import Queue
from _thread import *
import time

queue = Queue()

def webcam(queue):
    capture = cv2.VideoCapture(0)

    while True:
        ret, frame = capture.read()

        if ret == False:
            continue

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, imgencode = cv2.imencode('.jpg', frame, encode_param)

        data = numpy.array(imgencode)
        stringData = data.tobytes()

        queue.put(stringData)

        cv2.imshow('CLIENT', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

TCP_IP = '192.168.0.5'
TCP_PORT = 8000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))
print('connect')

start_new_thread(webcam, (queue,))

while True:
    stringData = queue.get()
    #time.sleep(1)
    sock.send(str(len(stringData)).ljust(16).encode())
    sock.send(stringData)
    print('send image')

sock.close()
cv2.destroyAllWindows()