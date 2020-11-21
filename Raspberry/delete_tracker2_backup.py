#Client
import socket
import cv2
import numpy as np
from queue import Queue
from _thread import *
import _thread
import time
import math
from sympy import Symbol, solve

capture = cv2.VideoCapture('test_cut.mp4')
resolution = (600,800)

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
tracker_arr = []

#GPS list
gps_list = []
gps_list.append((35.832909,128.754458)) # Left Up
gps_list.append((35.832842,128.754476)) # Right Up
gps_list.append((35.832850,128.754155)) # Left Down
gps_list.append((35.832776,128.754171)) # Right Down

#픽셀 간 위도 경도 비
gps_width = (math.sqrt((gps_list[0][0]-gps_list[1][0])**2+(gps_list[0][1]-gps_list[1][1])**2)+math.sqrt((gps_list[2][0]-gps_list[3][0])**2+(gps_list[2][1]-gps_list[3][1])**2))/2
gps_length = (math.sqrt((gps_list[0][0]-gps_list[2][0])**2+(gps_list[0][1]-gps_list[2][1])**2)+math.sqrt((gps_list[1][0]-gps_list[3][0])**2+(gps_list[1][1]-gps_list[3][1])**2))/2
pixel_width = resolution[0]
pixel_length = resolution[1]
width_rate = gps_width / pixel_width
length_rate = gps_length / pixel_length
#사람 발끝 좌표 * rate = gps 좌표

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
detect_count = 0

#rect.append((249, 444, 101, 202))
#rect.append((173, 550, 81, 161))
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
det_p = []
com_p = []
rect = ()
lock1 = _thread.allocate_lock()
lock2 = _thread.allocate_lock()
lock3 = _thread.allocate_lock()
lock4 = _thread.allocate_lock()
lock_cnt = _thread.allocate_lock()

start_trc = 0
cnt = 0
del_state = 0
del_num = 0

'''
def del_tracker(num):
    state = -1
    #print('num:',num,' state:',state)
    while state != num:
        lock4.acquire()
        pre_x = trc_p[2*num][0]
        pre_y = trc_p[2*num][1]
        time.sleep(1)
        cur_x = trc_p[2*num][0]
        cur_y = trc_p[2*num][1]
            
        distance = math.sqrt((pre_x-cur_x)**2+(pre_y-cur_y)**2)
        print('distance:',distance,' [',num,']')

        if distance < 10:
            print('hey')
            state = num
        lock4.release()
'''

def tracking(tracker, num):
    global start_trc
    global point
    global del_state
    global del_num
    global com_p
    global cnt
    global tracker_arr
    start_trc = 1
    count = 0
    cnt2 = 0
    gap_x = 0
    gap_y = 0
    timer = 0
    del_cnt = 1
    com_p.append(0)
    while com_p[num]!=1:
      point_tmp = ""
      
      #count += 1
      # read frame from video

      
      ret, img = capture.read()

      #img = cv2.resize(img, resolution)

      img = cv2.transpose(img)
      img = cv2.flip(img, 1)
      
      #img=perspective(img_pst)

      
      if not ret:
        exit()

      # update tracker and get position from new frame
      lock1.acquire()

      if del_cnt == del_state:
          if del_num < num:
              num -= 1
              del com_p[del_num]
          del_cnt += 1
          
      
      success, box = tracker.update(img)
      # if success:
      left, top, w, h = [int(v) for v in box]
      '''
      if cnt2 == 0:
        gap_x = rect[0] - left
        gap_y = rect[1] - top
        cnt2 += 1
      left = left + gap_x
      top = top + gap_y
      '''
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
        #start_new_thread(del_tracker, (num,))
        count=count+1

      else:
        trc_p[2*num]=pt1
        trc_p[2*num+1]=pt2
        #print(trc_p)
       
        if timer == 0:
            pre_x = trc_p[2*num][0]
            pre_y = trc_p[2*num][1]
            #print('timer0')
        elif timer == 10:
            cur_x = trc_p[2*num][0]
            cur_y = trc_p[2*num][1]
            #print('timer10: ',pre_x,',',cur_x)
            distance = math.sqrt((pre_x-cur_x)**2+(pre_y-cur_y)**2)
            #print('distance:',distance,' num:',num)
            if distance < 10:
                #print('delete! num:',num)
                del_state += 1
                del_num = num
                com_p[del_num]=1
                del trc_p[2*del_num]
                del trc_p[2*del_num]
                lock_cnt.acquire()
                cnt -= 1
                lock_cnt.release()
                #print(trc_p)
                
            timer=-1
        timer += 1 

      i=0
      #print(trc_p)
      length = len(trc_p)/2
      #print(length)
      for i in range(int(length)):
        cv2.rectangle(img, trc_p[2*i], trc_p[2*i+1], (255, 255, 255), 3)
        foot_x = int((trc_p[2*i][0]+trc_p[2*i+1][1])/2)
        foot_y = trc_p[2*i+1][1]
        point_tmp += str(foot_x) + ' ' + str(foot_y) + ' '

      point = point_tmp
      #print(point)
      lock1.release()
      cv2.imshow('img', img)

      lock2.acquire()
      if cv2.waitKey(1) == ord('q'):
          break
      lock2.release()
    del tracker_arr[num]
    
#Detect Function
def detect(gray, frame):
    
    global point
    global detect_count
    global det_p
    global cnt
    global rect
    detect_count = 0
    #point_tmp = ""
    #detected = cascade.detectMultiScale(gray, 1.3, 5)
    detected, _ = hog.detectMultiScale(frame)

    det_p=[]
    for (x, y, w, h) in detected:
        cv2.rectangle(frame, (x, y, w, h), (0, 255, 0), 3)
        #print('det =',(x, y, w, h))
        detect_count=detect_count+1
        #point_tmp += str(x)+' '+str(y)+' '
        
        left = x
        top = y
        right = left+w
        bottom = top+h

        pt1 = (int(left), int(top))
        pt2 = (int(right), int(bottom))
        
        det_p.append(pt1)
        det_p.append(pt2)

    #print(det_p)
    det_len = len(det_p)/2
    trc_len = len(trc_p)/2

    i=0
    j=0
    if int(trc_len) == 0:
        for i in range(int(det_len)):
            x = det_p[2*i][0]
            y = det_p[2*i][1]
            w = det_p[2*i+1][0] - det_p[2*i][0]
            h = det_p[2*i+1][1] - det_p[2*i][1]

            rect = (x, y, w, h)
            #print('rect = ', rect)
            lock_cnt.acquire()
            print('test0')
            print(tracker_arr)
            print(cnt)
            tracker_arr.append(OPENCV_OBJECT_TRACKERS['csrt']())
            tracker_arr[cnt].init(frame, rect)
            start_new_thread(tracking, (tracker_arr[cnt], cnt, ))
            lock_cnt.release()
            #print(tracker)
            #print(count)
            #print(len(tracker))
            cnt = cnt+1


    else:
        for j in range(int(det_len)):
            state = 0
            mean_x = (det_p[2*j+1][0]+det_p[2*j][0])/2
            mean_y = (det_p[2*j+1][1]+det_p[2*j][1])/2

            k=0
            for k in range(int(trc_len)):
                left_top_x = trc_p[2*k][0]
                left_top_y = trc_p[2*k][1]
                right_bottom_x = trc_p[2*k+1][0]
                right_bottom_y = trc_p[2*k+1][1]
                #print('det =', (mean_x, mean_y))
                #print('trc =', trc_p)

                if left_top_x < mean_x and left_top_y < mean_y and mean_x < right_bottom_x and mean_y < right_bottom_y:
                        state = 1
                        #print('test2')

            if state == 0:
                x = det_p[2*j][0]
                y = det_p[2*j][1]
                w = det_p[2*j+1][0] - det_p[2*j][0]
                h = det_p[2*j+1][1] - det_p[2*j][1]
                        
                rect = (x, y, w, h)
                #print(rect)
                lock_cnt.acquire()
                print('test1')
                print(tracker_arr)
                print(cnt)
                tracker_arr.append(OPENCV_OBJECT_TRACKERS['csrt']())
                tracker_arr[cnt].init(frame, rect)
                start_new_thread(tracking, (tracker_arr[cnt], cnt))
                lock_cnt.release()
                cnt = cnt+1
                #print('test3')

        #print('det = ', det_p)
        #print(len(det_p))

        #rect.append((x,y,w,h))
        #tracker.append(OPENCV_OBJECT_TRACKERS['csrt']())
        #tracker[count].init(img, rect[count])
        #count++
        
        #print(point_tmp)

    #print((x, y, w, h))
    #point = point_tmp
    
    return frame

def perspective(process):
    height, width = process.shape[:2]

    pts1 = np.float32([list(point_list[0]),list(point_list[1]),list(point_list[2]),list(point_list[3])])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

    #print(pts1)
    #print(pts2)

    M = cv2.getPerspectiveTransform(pts1,pts2)
    pst_img = cv2.warpPerspective(process, M, (width,height))

    return pst_img


#Show webcam
def webcam(queue):
    while True:
        ret, frame = capture.read()

        #frame=cv2.resize(frame, resolution)
        
        frame=cv2.transpose(frame)
        frame=cv2.flip(frame,1)
        
        #print(time)
        if ret == False:
            continue

        #encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        #result, imgencode = cv2.imencode('.jpg', process, encode_param)        
        #data = numpy.array(?)
        #stringData = data.tobytes()
        #queue.put(stringData)
        cv2.imshow('CLIENT', frame)
        #frame = perspective(frame)
        #cv2.imshow('perspective',frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detect_img=detect(gray, frame)
        cv2.imshow('test', detect_img)

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
#webcam(queue)

'''
while True:
    length = len(tracker)
    print(length)
    if int(length) == 0:
        lock.acquire()
        i=0
        j=0
        det_len = len(det_p)/2
        trc_len = len(trc_p)/2
        count = 0
        rect = ()

        
        #print(det_len)
        for i in range(int(det_len)):
            x = det_p[2*i][0]
            y = det_p[2*i][1]
            w = det_p[2*i+1][0] - det_p[2*i][0]
            h = det_p[2*i+1][1] - det_p[2*i][1]

            rect = (x, y, w, h)
            tracker.append(OPENCV_OBJECT_TRACKERS['csrt']())
            tracker[count].init(img, rect)
            start_new_thread(tracking, (tracker[count], count,))
            #print(rect)
            #print(count)
            #print(len(tracker))
            count = count+1

        lock.release()
    
'''

while True:
    #stringData = queue.get()
    #time.sleep(1)
    #sock.send(str(len(stringData)).ljust(16).encode())

    if point != "":
        sock.send(point.encode())
        print('send point')
        print(point)
        point = ""
    time.sleep(1)


sock.close()
cv2.destroyAllWindows()
