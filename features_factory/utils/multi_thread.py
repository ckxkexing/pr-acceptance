####################
# 多线程部分
####################
import queue
import random
import time

from vthread import pool

from utils.mysql_link import run_sql_mysqlclient

ls = queue.Queue()
producer = "pr"
consumer = "co"


@pool(100, gqueue=producer)
def creater(query, id):
    commits = run_sql_mysqlclient(query)
    ls.put({"query": query, "answer": commits})


@pool(1, gqueue=consumer)
def coster():
    # 这里之所以使用 check_stop 是因为，这里需要边生产边消费
    while not pool.check_stop(gqueue=producer):
        time.sleep(random.random() * 10)  # 随机睡眠 0.0 ~ 1.0 秒
        # pp = [ls.get() for _ in range(ls.qsize())]
        print("当前完成的个数: {}".format(ls.qsize()))


def duo_xian_cheng(cmds):
    for i, cmd in enumerate(cmds):
        creater(cmd, i)
    coster()  # 写作逻辑限制了这里的数量
    pool.wait(gqueue=producer)  # 等待默认的 gqueue=producer 组线程池全部停止再执行后面内容
    pool.wait(gqueue=consumer)  # 等待默认的 gqueue=consumer 组线程池全部停止再执行后面内容
    pp = [ls.get() for _ in range(ls.qsize())]
    return pp
