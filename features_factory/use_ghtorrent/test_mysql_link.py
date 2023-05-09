import pymysql
import datetime

host = '127.0.0.1'
port = 3306
db = 'ghtorrent_restore_2021'
user = 'blf'
password = '123456'

def get_connection():
    conn = pymysql.connect(host=host, port=port, db=db, user=user, password=password)
    return conn

def check():

    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("select * from users limit 1")
    data = cursor.fetchone()
    print(data)
    cursor.close()
    conn.close()

def run_sql(sql_code):
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql_code)
    data = cursor.fetchall()
    # print(data)
    cursor.close()
    conn.close()
    return data

# 使用mysqlclient，据说速度是10x
import MySQLdb
import MySQLdb.cursors

def get_connection_mysqlclient():
    conn = MySQLdb.connect(
        host=host,
        port = port,
        user=user,
        passwd=password,
        db =db
        )
    return conn

def check_mysqlclient():
    conn = get_connection_mysqlclient()
    cursor = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)
    cursor.execute("select * from users limit 1")
    data = cursor.fetchone()
    print(data)
    cursor.close()
    conn.close()

def run_sql_mysqlclient(sql_code):
    conn = get_connection_mysqlclient()
    cursor = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)
    cursor.execute(sql_code)
    data = cursor.fetchall()
    # print(data)
    cursor.close()
    conn.close()
    return data

import time
class Timer():
    def __init__(self):
        self.time = time.time()

    def start(self):
        self.time = time.time()

    def stop(self):
        return time.time() - self.time

if __name__ == '__main__':
    # print(run_sql("select id from users where login='whoistosch';"))
    # print(run_sql("select * from followers where user_id='1';"))
    
    timer = Timer()
    # followers = run_sql_mysqlclient(f"select * from followers where user_id=100;")
    # print(f"run_sql_mysqlclient{timer.stop()}")
    followers = run_sql(f"select * from followers where user_id=100;")
    print(f"run_sql{timer.stop()}")
    pr_date = datetime.datetime.strptime('2021-04-18T08:21:20Z', "%Y-%m-%dT%H:%M:%SZ")
    cnt = 0
    print(len(followers))
    print(followers)
    for event in followers:
        # print(event['created_at'])
        print(event)
        if event['created_at'] < pr_date:
            cnt+=1
    print(cnt)
