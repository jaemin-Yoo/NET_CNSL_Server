#Client
import socket
import cv2
import numpy as np
from queue import Queue
from _thread import *
import _thread
import time

capture = cv2.VideoCapture('test_cut.mp4')
ret, img_tmp = capture.read()
img = cv2.transpose(img_tmp)
img = cv2.flip(img, 1)

# initialize tracker
OPENCV_OBJECT_TRACKERS = {
  "csrt": cv2.TrackerCSRT_create,
  "kcf": cv2.TrackerKCF_create,
  "boosting": cv2.TrackerBoosting_create,
  "mil": cv2.TrackerMIL_create,
  "tld": cv2.TrackerTLD_create,
  "medianflow": cv2.TrackerMedianFlow_create,
  "mosse": cv2.TrackerMOSSE_create
}

# global variables
top_bottom_list, left_right_list = [], []
#rect = []
tracker = []

#GPS list
gps_list = []
gps_list.append((35.832909,128.754458)) # Left Up
gps_list.append((35.832842,128.754476)) # Right Up
gps_list.append((35.832850,128.754155)) # Left Down
gps_list.append((35.832776,128.754171)) # Right Down

#픽셀 간 위도 경도 비
width=480
length=852
lat_mean=((gps_list[0][0]-gps_list[1][0])+(gps_list[2][0]-gps_list[3][0]))/2
long_mean=((gps_list[0][1]-gps_list[2][1])+(gps_list[1][1]-gps_list[3][1]))/2
lat_pixel=lat_mean/width
long_pixel=long_mean/length

#Perspective Point
point_list = []
point_list.append((201,296)) # Left Up
point_list.append((340,303)) # Right Up
point_list.append((69,529)) # Left Down
point_list.append((391,533)) # Right Down

#Detect Person .xml
#cascadefile = "haarcascade_frontalface_default.xml"
#cascade = cv2.CascadeClassifier(cascadefile)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#Initialize
queue = Queue()
point = ""

#rect.append((198, 463, 20, 65))
#rect.append((305, 466, 24, 53))
#rect.append((239, 413, 20, 52))
#rect.append((287, 392, 20, 48))

#tracker.append(OPENCV_OBJECT_TRACKERS['csrt']())
#tracker.append(OPENCV_OBJECT_TRACKERS['csrt']())
#tracker.append(OPENCV_OBJECT_TRACKERS['csrt']())
#tracker.append(OPENCV_OBJECT_TRACKERS['csrt']())

#tracker[0].init(img, rect[0])
#tracker[1].init(img, rect[1])
#tracker[2].init(img, rect[2])
#tracker[3].init(img, rect[3])

trc_p = []

lock = _thread.allocate()

def tracking(tracker, num):
    count = 0
    while True:
      #count += 1
      # read frame from video
      lock.acquire()
      ret, img_tmp = capture.read()
      img = cv2.transpose(img_tmp)
      img = cv2.flip(img, 1)

      if not ret:
        exit()

      # update tracker and get position from new frame
      success, box = tracker.update(img)
      # if success:
      left, top, w, h = [int(v) for v in box]
      right = left + w
      bottom = top + h

      # save sizes of image
      top_bottom_list.append(np.array([top, bottom]))
      left_right_list.append(np.array([left, right]))

      # use recent 10 elements for crop (window_size=10)
      if len(top_bottom_list) > 10:
        del top_bottom_list[0]
        del left_right_list[0]

      # compute moving average
      avg_height_range = np.mean(top_bottom_list, axis=0).astype(np.int)
      avg_width_range = np.mean(left_right_list, axis=0).astype(np.int)
      avg_center = np.array([np.mean(avg_width_range), np.mean(avg_height_range)]) # (x, y)

      # compute scaled width and height
      scale = 1.3
      avg_height = (avg_height_range[1] - avg_height_range[0]) * scale
      avg_width = (avg_width_range[1] - avg_width_range[0]) * scale

      # compute new scaled ROI
      avg_height_range = np.array([avg_center[1] - avg_height / 2, avg_center[1] + avg_height / 2])
      avg_width_range = np.array([avg_center[0] - avg_width / 2, avg_center[0] + avg_width / 2])
      '''
      # fit to output aspect ratio
      if fit_to == 'width':
          avg_height_range = np.array([
            avg_center[1] - avg_width * output_size[1] / output_size[0] / 2,
            avg_center[1] + avg_width * output_size[1] / output_size[0] / 2
          ]).astype(np.int).clip(0, 9999)

          avg_width_range = avg_width_range.astype(np.int).clip(0, 9999)
        elif fit_to == 'height':
          avg_height_range = avg_height_range.astype(np.int).clip(0, 9999)

          avg_width_range = np.array([
            avg_center[0] - avg_height * output_size[0] / output_size[1] / 2,
            avg_center[0] + avg_height * output_size[0] / output_size[1] / 2
          ]).astype(np.int).clip(0, 9999)

        # crop image
        result_img = img[avg_height_range[0]:avg_height_range[1], avg_width_range[0]:avg_width_range[1]].copy()

      # resize image to output size
      result_img = cv2.resize(result_img, output_size)
      '''
      # visualize
      pt1 = (int(left), int(top))
      pt2 = (int(right), int(bottom))
      pt = [pt1, pt2]
      #print(pt)

      if count < 1:
        trc_p.append(pt1)
        trc_p.append(pt2)
        count=count+1

      else:
        trc_p[2*num]=pt1
        trc_p[2*num+1]=pt2

      i=0
      
      #print('trc_p', trc_p)
      length = len(trc_p)/2
      #print(length)
      for i in range(int(length)):
        cv2.rectangle(img, trc_p[2*i], trc_p[2*i+1], (255, 255, 255), 3)

      lock.release()

      cv2.imshow('img', img)

      lock.acquire()
      key = cv2.waitKey(1)
      if key == ord('q'):
          break
      lock.release()

t_cnt = 0
#Detect Function
def detect(gray, frame):
    global point
    global t_cnt
    count = 0
    point_tmp = ""
    #detected = cascade.detectMultiScale(gray, 1.3, 5)
    detected, _ = hog.detectMultiScale(frame)
    
    for (x, y, w, h) in detected:
        cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)
        point_tmp += str(x)+' '+str(y)+' '
        print((x, y, w, h))
        #rect.append((x,y,w,h))
        #tracker.append(OPENCV_OBJECT_TRACKERS['csrt']())
        #tracker[count].init(img, rect[count])
        #count++
        
        #print(point_tmp)

    i=0
    if t_cnt == 0:
        for i in range(1):
            rect = (64, 504, 67, 134)
            #print(rect)
            tracker.append(OPENCV_OBJECT_TRACKERS['csrt']())
            tracker[i].init(img, rect)
            start_new_thread(tracking, (tracker[i], i,))
            t_cnt += 1
     
    point = point_tmp
    return frame

def perspective(process):
    height, width = process.shape[:2]

    pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    #print(pts1)
    #print(pts2)

    M = cv2.getPerspectiveTransform(pts1,pts2)
    img_result = cv2.warpPerspective(process, M, (width,height))

    #gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
    #process=detect(gray, img_result)

    cv2.imshow('test', process)
    return process


#Show webcam
def webcam(queue):
    #time = 0
    while True:
        ret, frame_rotate = capture.read()
        
        frame=cv2.transpose(frame_rotate)
        frame=cv2.flip(frame,1)
        
        
        if ret == False:
            continue

        #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        #result, imgencode = cv2.imencode('.jpg', process, encode_param)        
        #data = numpy.array(?)
        #stringData = data.tobytes()
        #queue.put(stringData)

        cv2.imshow('CLIENT', frame)

        #if time > 100:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        process=detect(gray, frame)
        cv2.imshow('test', process)

        #time += 1
        #process=perspective(frame)
        #print('test')
        
        # initialize tracker
        #rect = (195, 464, 27, 64)
        #tracker.init(process, rect)
        #tracking(process)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

TCP_IP = '10.100.201.132'
TCP_PORT = 10002

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))
print('connect')


start_new_thread(webcam, (queue,))
#start_new_thread(tracking, (tracker[0], 0,))
#start_new_thread(tracking, (tracker[1], 1,))
#start_new_thread(tracking, (tracker[2], 2,))
#start_new_thread(tracking, (tracker[3], 3,))


#webcam(queue)


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
