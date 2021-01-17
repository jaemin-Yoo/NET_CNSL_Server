#Client
import socket
import cv2
import numpy as np
from queue import Queue
from _thread import *
import time
from sympy import Symbol, solve
import math

resolution = (800, 800)
img = 'tt3.mp4'

#GPS list
gps_list = []

#Right up 0,0
gps_list.append((35.832909,128.754458)) # Left Up
gps_list.append((35.832842,128.754476)) # Right Up
gps_list.append((35.832850,128.754155)) # Left Down
gps_list.append((35.832776,128.754171)) # Right Down
'''
#Left down 0,0
gps_list.append((35.832776,128.754171)) # Left Up
gps_list.append((35.832850,128.754155)) # Right Up
gps_list.append((35.832842,128.754476)) # Left Down
gps_list.append((35.832909,128.754458)) # Right Down
'''
#픽셀 간 GPS거리 비
gps_width = (math.sqrt((gps_list[0][0]-gps_list[1][0])**2+(gps_list[0][1]-gps_list[1][1])**2)+math.sqrt((gps_list[2][0]-gps_list[3][0])**2+(gps_list[2][1]-gps_list[3][1])**2))/2
gps_length = (math.sqrt((gps_list[0][0]-gps_list[2][0])**2+(gps_list[0][1]-gps_list[2][1])**2)+math.sqrt((gps_list[1][0]-gps_list[3][0])**2+(gps_list[1][1]-gps_list[3][1])**2))/2
pixel_width = resolution[0]
pixel_length = resolution[1]
x_rate = gps_width / pixel_width
y_rate = gps_length / pixel_length

#Perspective Point
point_list = []
def mouse_callback(event, x, y, flags, param):
    global point_list, count, img_original


    # 마우스 왼쪽 버튼 누를 때마다 좌표를 리스트에 저장
    if event == cv2.EVENT_LBUTTONDOWN:
        print("(%d, %d)" % (x, y))
        point_list.append((x, y))

        print(point_list)
        cv2.circle(img_original, (x, y), 3, (0, 0, 255), -1)



cv2.namedWindow('original')
cv2.setMouseCallback('original', mouse_callback)


# 원본 이미지
capture = cv2.VideoCapture(img)
ret, img_original = capture.read()
img_original = cv2.resize(img_original, (resolution[0],resolution[1]))


while(True):

    cv2.imshow("original", img_original)


    height, width = img_original.shape[:2]


    if cv2.waitKey(1)&0xFF == 32: # spacebar를 누르면 루프에서 빠져나옵니다.
        break
cv2.destroyAllWindows()


#Detect Person .xml
#cascadefile = "haarcascade_frontalface_default.xml"
#cascade = cv2.CascadeClassifier(cascadefile)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Initialize
queue = Queue()
point = ""

def gps_conversion(foot_x, foot_y):
    a = Symbol('a')
    b = Symbol('b')

    equation1 = (gps_list[0][1]-a)**2+(gps_list[0][0]-b)**2-(foot_x*x_rate)**2
    equation2 = ((gps_list[0][0]-gps_list[1][0])/(gps_list[0][1]-gps_list[1][1]))*(a-gps_list[0][1])+gps_list[0][0]-b
    res=solve((equation1, equation2), dict=True)
    #print(res[1])

    x = Symbol('x')
    y = Symbol('y')

    equation1 = (res[1][a]-x)**2+(res[1][b]-y)**2-(foot_y*y_rate)**2
    equation2 = (-(gps_list[0][1]-gps_list[1][1])/(gps_list[0][0]-gps_list[1][0]))*(x-res[1][a])+res[1][b]-y
    res=solve((equation1, equation2), dict=True)
    gps = str(res[0][y])+' '+str(res[0][x])+' '
    return gps

#Detect Function
def detect(gray, frame):
    global point
    point_tmp = ""
    #detected = cascade.detectMultiScale(gray, 1.3, 5)
    detected, _ = hog.detectMultiScale(frame)
    
    for (x, y, w, h) in detected:
        cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)
        foot_x = x+(w/2)
        foot_y = y+(h-60)
        point_tmp += gps_conversion(foot_x, foot_y)
        print(point_tmp)

    point = '0 '+point_tmp
    return frame

def perspective(process):
    height, width = process.shape[:2]

    pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    #print(pts1)
    #print(pts2)

    M = cv2.getPerspectiveTransform(pts1,pts2)
    img_result = cv2.warpPerspective(process, M, (width,height))

    return img_result


#Show webcam
prev_time = 0
FPS = 24
def webcam(queue):
    global prev_time
    capture = cv2.VideoCapture(img)
    cnt = -1
    current_time = 0
    wait = 0
    while True:
        #start = time.time()
        ret, frame = capture.read()
        frame = cv2.resize(frame, (resolution[0],resolution[1]))
        current_time = time.time() - prev_time
        if current_time < 1 and cnt==-1:
            wait = current_time/(1/FPS)
        #print(current_time)
        cnt+=1
        if (ret is True) and (current_time > 1./ FPS) and (cnt==int(wait)):

            prev_time = time.time()
            cnt = -1
        
            #frame=cv2.transpose(frame)
            #frame=cv2.flip(frame,1)
            
            
            if ret == False:
                continue

            #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            #result, imgencode = cv2.imencode('.jpg', process, encode_param)        
            #data = numpy.array(?)
            #stringData = data.tobytes()
            #queue.put(stringData)

            cv2.imshow('CLIENT', frame)
            frame=perspective(frame)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            process=detect(gray, frame)

            cv2.imshow('test', process)
            #end = time.time()
            #diff = end-start
            #print(diff)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

#TCP_IP = '10.100.201.132'
#TCP_PORT = 10002

TCP_IP = '127.0.0.1'
TCP_PORT = 8000

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))
print('connect')

        

start_new_thread(webcam, (queue,))
#webcam(queue)


while True:
    #stringData = queue.get()
    #time.sleep(1)
    #sock.send(str(len(stringData)).ljust(16).encode())

    if point != "0 ":
        sock.send(point.encode())
        print('send point')
        point = "0 "

    #time.sleep(0.5)

sock.close()
cv2.destroyAllWindows()
