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

lock = threading.Lock()
#db_conn = pymysql.connect(host='116.89.189.36', user='root', passwd='4556',
#                          db='location', charset='utf8')
#curs = db_conn.cursor()

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


def threaded(conn, addr, queue, array):
    cnt = 1
    i=0
    j=0
    print('Connect by :', addr[0], ':', addr[1])
    
    while True:                
        start = time.time()
        sql_sel = "select ip from person where ip=%s"
        var_sel = (addr[0]+'%')
        curs.execute(sql_sel, var_sel)
        #rows = curs.rowcount

        #length = recvall(conn, 16)
        #stringData = recvall(conn, int(length))
        #data = numpy.frombuffer(stringData, dtype='uint8')
        data = conn.recv(1024)
        point = data.decode()
        location = point.split()
        print(point,' ',addr[0])
        #print(location)
        #decimg = cv2.imdecode(data, 1)
        #cv2.imshow('SERVER'+str(i), decimg)
        end = time.time()
        difftime = end - start
        #print(difftime)
        lock.acquire()
        #print(array[i],", time : ", format(difftime,'.6f'))
        lock.release()

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
            val_ins = (addr[0]+'/'+str(j), now, location[2*j], location[2*j+1])

            curs.execute(sql_ins, val_ins)
            db_conn.commit()


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
