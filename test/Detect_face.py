#Client
import matplotlib.pyplot as plt
import socket
import cv2
import numpy
from queue import Queue
from _thread import *
import time

'''
point_list = []
point_list.append((,)) # Left Up
point_list.append((,)) # Right Up
point_list.append((,)) # Left Down
point_list.append((,)) # Right Down
'''

temp = cv2.imread('test.jpg')
img_original = cv2.resize(temp, (800, 800)) # Picture size


#Detect Person .xml
cascadefile = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascadefile)

queue = Queue()
point = ""

#Detect Function
def detect(gray, frame):
    global point
    faces = cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)
        point = str(x)+' '+str(y)
        print(point)
        
    return frame

def webcam(queue):
    capture = cv2.VideoCapture('test.mp4')

    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        process=detect(gray, frame)

        
        if ret == False:
            continue

        #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        #result, imgencode = cv2.imencode('.jpg', process, encode_param)        
        #data = numpy.array(?)
        #stringData = data.tobytes()
        #queue.put(stringData)

        cv2.imshow('CLIENT', process)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

TCP_IP = '10.100.201.132'
TCP_PORT = 10002

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))
print('connect')

start_new_thread(webcam, (queue,))

while True:
    #stringData = queue.get()
    #time.sleep(1)
    #sock.send(str(len(stringData)).ljust(16).encode())

    if point != "":
        sock.send(point.encode())
        print('send point')
        point = ""
    time.sleep(1)


sock.close()
cv2.destroyAllWindows()
