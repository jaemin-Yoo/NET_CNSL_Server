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
lock1 = threading.Lock()
db_conn = pymysql.connect(host='116.89.189.36', user='root', passwd='4556',
                          db='location', charset='utf8')
curs = db_conn.cursor()
curs2 = db_conn.cursor()

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

temp =  []
def threaded(conn, addr, queue, array):
    global temp
    cnt = 1
    i=0
    j=0
    loc_tmp = []
    location = []
    state = 0
    print('Connect by :', addr[0], ':', addr[1])
    
    while True:
        min = 100
        start = time.time()

        num = 0
        while True:
            if array[num] == addr[1]:
                break
            else:
                num+=1

        temp[num] = loc_tmp
                
                
        data = conn.recv(1024)
        point = data.decode()

        lock.acquire()
        print('recv: ',point)
        location= point.split()
        loc_tmp = point.split()
        print(location)
        print(temp)

        cnt=0
        tmp_len = len(temp)
        print(int(tmp_len))
        for cnt in range(int(tmp_len)):
            if cnt != num and temp[cnt] != 0:
                i=0
                j=0
                loc_len = len(location)/2
                tem_len = len(temp[cnt])/2
                print(tem_len)
                for i in range(int(tem_len)):
                    for j in range(int(loc_len)):
                        tem_lat = float(temp[cnt][2*i])
                        tem_long = float(temp[cnt][2*i+1])
                        loc_lat = float(location[2*j])
                        loc_long = float(location[2*j+1])
                        dist = math.sqrt((tem_lat-loc_lat)**2+(tem_long-loc_long)**2)
                        print('dist: ',dist)
                        if (dist < 0.00003) and (dist < min):
                            min = dist
                            lat = (tem_lat+loc_lat)/2
                            long = (tem_long+loc_long)/2
                            location[2*j] = str(lat)
                            location[2*j+1] = str(long)
                            print('Merge!!')
                            print('reset point: ',(lat, long))
                        else:
                            state = 1
                    if state == 1:
                        location.append(temp[cnt][2*i])
                        location.append(temp[cnt][2*i+1])
                        state = 0
            temp[cnt]=0

        end = time.time()
        difftime = end - start

        now = datetime.now()
                
        sql_del = "delete from person where ip like %s"
        val_del = (addr[0]+'%')
        lock1.acquire()
        curs.execute(sql_del, val_del)
        db_conn.commit()
        len_tmp = len(location)
        length = len_tmp/2

        print('location: ',location)                    
        for j in range(int(length)):
            sql_ins = "insert into person values(%s, %s, %s, %s)"
            val_ins = (addr[0]+'/'+str(j), now, location[2*j], location[2*j+1])
            curs.execute(sql_ins, val_ins)
            db_conn.commit()
        lock1.release()

        lock.release()


server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((TCP_IP, TCP_PORT))
server_socket.listen()
print('Listening')

def delete_db():
    while True:
        lock1.acquire()
        sql_sel = "select * from person limit 1"
        curs2.execute(sql_sel)
        rows = curs2.fetchone()
        lock1.release()
        time.sleep(10)

        lock1.acquire()
        sql_sel2 = "select * from person limit 1"
        curs2.execute(sql_sel2)
        rows2 = curs2.fetchone()

        if (rows == rows2) and (rows2 is not None):
            sql_del = "delete from person"
            curs2.execute(sql_del)
            db_conn.commit()
            print('Time over Delete')
        lock1.release()

    
    
start_new_thread(delete_db, ())


array=[]

while True:
    conn, addr = server_socket.accept()
    array.append(addr[1])
    temp.append((0,0))
    print(array)
    start_new_thread(threaded, (conn, addr, enclosure_queue, array,))
    

server_socket.close()
#cv2.destroyAllWindows()
