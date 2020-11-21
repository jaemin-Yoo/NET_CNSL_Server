import pymysql
from datetime import datetime


now = datetime.now()
conn = pymysql.connect(host='116.89.189.36', user='root', password='4556',
                       db='location', charset='utf8')
curs = conn.cursor()



sql_sel = "select ip from person where ip=%s"
var_sel = ("192.168.0.5")
curs.execute(sql_sel, var_sel)
cnt = curs.rowcount

if cnt == 0:
    sql_ins = "insert into person values(%s, %s, %s, %s)"
    var_ins = ("192.168.0.5", now, "30", "127")

    curs.execute(sql_ins, var_ins)
    conn.commit()


else:
    sql_ins = "update person set time=%s, Latitude=%s, Longitude=%s where ip=%s"
    var_ins = (now, "32", "127", "192.168.0.5")

    curs.execute(sql_ins, var_ins)
    conn.commit()


print('Success')

