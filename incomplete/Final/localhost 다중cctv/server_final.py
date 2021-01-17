# Server
import socket
#import cv2
#import numpy
import time
from queue import Queue
from _thread import *
import threading
import pymysql
from datetime import datetime
import math

lock = threading.Lock()
db_conn = pymysql.connect(host='116.89.189.36', user='root', passwd='4556',
                          db='location', charset='utf8')
curs = db_conn.cursor()

'''
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

'''
TCP_IP = ''
TCP_PORT = 8000


enclosure_queue = Queue()

temp0 = []
temp1 = []
def threaded(conn, addr, queue, array):
    global temp0
    global temp1
    cnt = 1
    i=0
    j=0
    location = []
    print('Connect by :', addr[0], ':', addr[1])
    
    while True:                
        start = time.time()
        #sql_sel = "select ip from person where ip=%s"
        #var_sel = (addr[0]+'%')
        #curs.execute(sql_sel, var_sel)
        #rows = curs.rowcount

        #length = recvall(conn, 16)
        #stringData = recvall(conn, int(length))
        #data = numpy.frombuffer(stringData, dtype='uint8')
        if len(location) != 0:
            print(location)
            if location[0] == '0':
                temp0 = location
            elif location[0] == '1':
                temp1 = location
                
        data = conn.recv(1024)
        point = data.decode()

        lock.acquire()
        print(point)
        location = point.split()
        print('0: ',location[0],' ',temp1)
        print('1: ',location[0],' ',temp0)
        if location[0] == '0' and len(temp1) != 0:
            print('temp: ',temp1)
            print('loca: ',location)
            if temp1[0] != location[0]:
                i=0
                j=0
                loc_len = len(location)/2
                tem_len = len(temp1)/2
                for i in range(int(tem_len)):
                    for j in range(int(loc_len)):
                        tem_lat = float(temp1[2*i+1])
                        tem_long = float(temp1[2*i+2])
                        loc_lat = float(location[2*j+1])
                        loc_long = float(location[2*j+2])
                        dist = math.sqrt((tem_lat-loc_lat)**2+(tem_long-loc_long)**2)
                        print('dist: ',dist)
                        if dist < 0.00003:
                            lat = (tem_lat+loc_lat)/2
                            long = (tem_long+loc_long)/2
                            location[2*j+1] = str(lat)
                            location[2*j+2] = str(long)
                            print('!!!')
                            print((lat, long))

        elif location[0] == '1' and len(temp0) != 0:
            print('temp: ',temp0)
            print('loca: ',location)
            if temp0[0] != location[0]:
                i=0
                j=0
                loc_len = len(location)/2
                tem_len = len(temp0)/2
                for i in range(int(tem_len)):
                    for j in range(int(loc_len)):
                        tem_lat = float(temp0[2*i+1])
                        tem_long = float(temp0[2*i+2])
                        loc_lat = float(location[2*j+1])
                        loc_long = float(location[2*j+2])
                        dist = math.sqrt((tem_lat-loc_lat)**2+(tem_long-loc_long)**2)
                        print('dist: ',dist)
                        if dist < 0.00003:
                            lat = (tem_lat+loc_lat)/2
                            long = (tem_long+loc_long)/2
                            location[2*j+1] = str(lat)
                            location[2*j+2] = str(long)
                            print('!!!')
                            print((lat, long))

        
                
        #print(location)
        #decimg = cv2.imdecode(data, 1)
        #cv2.imshow('SERVER'+str(i), decimg)
        end = time.time()
        difftime = end - start
        #print(difftime)
        #lock.acquire()
        #print(array[i],", time : ", format(difftime,'.6f'))
        #lock.release()

        now = datetime.now()
                

        sql_del = "delete from person where ip like %s"
        val_del = (addr[0]+'%')

        curs.execute(sql_del, val_del)
        db_conn.commit()
        print(len(location))
        len_tmp = len(location)
        length = len_tmp/2
                    
        for j in range(int(length)):
            sql_ins = "insert into person values(%s, %s, %s, %s)"
            val_ins = (addr[0]+'/'+str(j), now, location[2*j+1], location[2*j+2])

            curs.execute(sql_ins, val_ins)
            db_conn.commit()

        lock.release()


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((TCP_IP, TCP_PORT))
server_socket.listen()
print('Listening')

array=[]

while True:
    conn, addr = server_socket.accept()
    array.append(addr[0])
    print(array)
    start_new_thread(threaded, (conn, addr, enclosure_queue, array,))
    

server_socket.close()
#cv2.destroyAllWindows()
