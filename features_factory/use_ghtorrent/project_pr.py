'''
     对项目的其他PR进行信息统计
'''
import os
import json
import pymysql
import datetime
from .utils import Timer
from .utils import write_csv_data
from .test_mysql_link import get_connection, run_sql
from .test_mysql_link import get_connection_mysqlclient, run_sql_mysqlclient
from tqdm import tqdm


####################
# 多线程部分
####################
from vthread import pool, lock
import time, random, queue
ls = queue.Queue()
producer = 'pr'
consumer = 'co'

@pool(100, gqueue=producer)
def creater(query, id):
    commits = run_sql_mysqlclient(query)
    ls.put({'query':query, 'answer': commits})

@pool(1, gqueue=consumer)
def coster():
    # 这里之所以使用 check_stop 是因为，这里需要边生产边消费
    while not pool.check_stop(gqueue=producer):
        time.sleep(random.random()*10) # 随机睡眠 0.0 ~ 1.0 秒
        # pp = [ls.get() for _ in range(ls.qsize())]
        print('当前完成的个数: {}'.format(ls.qsize()))

def duo_xian_cheng(cmds):
    for i, cmd in enumerate(cmds):
        creater(cmd, i)
    coster() # 写作逻辑限制了这里的数量
    pool.wait(gqueue=producer) # 等待默认的 gqueue=producer 组线程池全部停止再执行后面内容
    pool.wait(gqueue=consumer) # 等待默认的 gqueue=consumer 组线程池全部停止再执行后面内容
    pp = [ls.get() for _ in range(ls.qsize())]
    return pp

# 4.3
# PR提交前commits个数
def tongji_project_before_pr_commits(json_file, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_name = 'apache/spark'
        project_name_all > 'apache_spark'
    '''
    job = 'project_commits'
    timer = Timer()
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr']    = data['url']
            tmp['login'] = data['user']['login']
            tmp['date']  = data['created_at']
            pulls.append(tmp)

    print(f"加载pulls用时{timer.stop()}")
    res_header = ['pr_url', 'login', f'before_pr_{job}']
    res = []
    repo_name = repo_name.split('/')
    owner = repo_name[0]
    project_name = repo_name[1]
    owner_sq = f"select id from users where login = '{owner}';"
    owner_id = run_sql_mysqlclient(owner_sq)
    project_sq = f"select id from projects where owner_id = {owner_id[0]['id']} and name = '{project_name}';" # 查出owner_id
    project_id = run_sql_mysqlclient(project_sq)
    date_sq = f"select created_at from commits where project_id = {project_id[0]['id']};"
    project = run_sql_mysqlclient(date_sq)
    idx, cnt = 0, 0
    pulls.reverse()
    
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])

        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        # 0000-00-00 00:00:00 => means time is undefined
        while idx < len(project) and (isinstance(project[idx]['created_at'], str) or project[idx]['created_at']==None):
            idx += 1
        while idx < len(project) and pr_date > project[idx]['created_at']:
            cnt += 1
            idx += 1
        cur.append(cnt)
        res.append(cur)
    # 13845s 4h左右
    print(f"计算pulls的followers用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_3.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")


# 4.4
# pr提交前一个月commits个数
def tongji_project_before_pr_commits_in_months(json_file, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_name = 'apache/spark'
        project_name_all > 'apache_spark'
    '''
    job = 'project_commits_in_month'
    timer = Timer()
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr'] = data['url']
            tmp['login'] = data['user']['login']
            tmp['date'] = data['created_at']
            pulls.append(tmp)

    print(f"加载pulls用时{timer.stop()}")

    res_header = ['pr_url', 'login', f'before_pr_{job}']
    res = []
    repo_name = repo_name.split('/')
    owner = repo_name[0]
    project_name = repo_name[1]
    owner_sq = f"select id from users where login = '{owner}';"
    owner_id = run_sql_mysqlclient(owner_sq)
    project_sq = f"select id from projects where owner_id = {owner_id[0]['id']} and name = '{project_name}';" # 查出owner_id
    project_id = run_sql_mysqlclient(project_sq)
    date_sq = f"select created_at from commits where project_id = {project_id[0]['id']};"
    project = run_sql_mysqlclient(date_sq)
    le, ri = 0, 0
    pulls.reverse()
    datedelta = datetime.timedelta(days=31)
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])

        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        while ri < len(project) and (isinstance(project[ri]['created_at'], str) or project[ri]['created_at']==None):
            ri += 1
        while ri < len(project) and pr_date > project[ri]['created_at']:
            ri += 1
        
        while le < len(project) and (isinstance(project[le]['created_at'], str) or project[le]['created_at']==None):
            le += 1
        while le < len(project) and pr_date - datedelta > project[le]['created_at']:
            le += 1

        cur.append(ri - le)
        res.append(cur)
    print(f"计算 [ pr提交前一个月commits个数 ] 用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_4.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")

# 4.5
# pr提交前的comment数目
def tongji_project_before_pr_comments(json_file, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_name = 'apache/spark'
        project_name_all > 'apache_spark'
    '''
    job = 'project_pr_comments'
    timer = Timer()
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr'] = data['url']
            tmp['login'] = data['user']['login']
            tmp['date'] = data['created_at']
            pulls.append(tmp)

    print(f"加载pulls用时{timer.stop()}")

    # 5s
    res_header = ['pr_url', 'login', f'before_pr_{job}']
    res = []
    repo_name = repo_name.split("/")
    owner = repo_name[0]
    project_name = repo_name[1]
    owner_sq = f"select id from users where login = '{owner}';"
    #print(owner_sq)
    owner_id = run_sql_mysqlclient(owner_sq)
    project_sq = f"select id from projects where owner_id = {owner_id[0]['id']} and name = '{project_name}';" # 查出owner_id
    project_id = run_sql_mysqlclient(project_sq)
    date_sq = f"select pull_request_id from issues where repo_id = {project_id[0]['id']};"
    pull_request = run_sql_mysqlclient(date_sq)

    pulls.reverse()
    comments = []
    # datedelta = datetime.timedelta(days=31)
    for idx in pull_request:
        if idx['pull_request_id'] == None:
            continue
        comment_sq = f"select * from pull_request_comments where pull_request_id = {idx['pull_request_id']};"
        comment =  list(run_sql_mysqlclient(comment_sq))
        comments.extend(comment)

    comments = sorted(comments, key=lambda x:x['created_at'])
    # print(comments)

    idx, cnt = 0, 0
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        
        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        while idx < len(comments) and (isinstance(comments[idx]['created_at'], str) or comments[idx]['created_at']==None):
            idx += 1
        while idx < len(comments) and pr_date > comments[idx]['created_at']:
            idx += 1
            cnt += 1

        cur.append(cnt)    
        res.append(cur)
        
    print(f"计算 [pr提交前的comment数目] 用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_5.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"写入入用时{timer.stop()}")

# 4.6
# PR提交前一个月的comment数目
def tongji_project_before_pr_comments_in_months(json_file, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_name = 'apache/spark'
        project_name_all > 'apache_spark'
    '''
    job = 'project_comments_in_month'
    timer = Timer()
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr'] = data['url']
            tmp['login'] = data['user']['login']
            tmp['date'] = data['created_at']
            pulls.append(tmp)

    print(f"加载pulls用时{timer.stop()}")

    res_header = ['pr_url', 'login', f'before_pr_{job}']
    res = []
    repo_name = repo_name.split("/")
    owner = repo_name[0]
    project_name = repo_name[1]
    owner_sq = f"select id from users where login = '{owner}';"
    #print(owner_sq)
    owner_id = run_sql_mysqlclient(owner_sq)
    project_sq = f"select id from projects where owner_id = {owner_id[0]['id']} and name = '{project_name}';" # 查出owner_id
    project_id = run_sql_mysqlclient(project_sq)
    date_sq = f"select pull_request_id from issues where repo_id = {project_id[0]['id']};"
    pull_request = run_sql_mysqlclient(date_sq)

    pulls.reverse()
    comments = []
    datedelta = datetime.timedelta(days=31)

    for idx in pull_request:
        # print(idx['pull_request_id'])
        if idx['pull_request_id'] == None:
            continue
        # code review comments
        comment_sq = f"select * from pull_request_comments where pull_request_id = {idx['pull_request_id']};"
        comment =  run_sql_mysqlclient(comment_sq)
        comments.extend(comment)

    comments = sorted(comments, key=lambda x:x['created_at'])

    le, ri = 0, 0
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        
        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        while ri < len(comments) and (isinstance(comments[ri]['created_at'], str) or comments[ri]['created_at']==None):
            ri += 1
        while ri < len(comments) and pr_date > comments[ri]['created_at']:
            ri += 1

        while le < len(comments) and (isinstance(comments[le]['created_at'], str) or comments[le]['created_at']==None):
            le += 1
        while le < len(comments) and pr_date - datedelta > comments[le]['created_at']:
            le += 1

        cur.append(ri-le)
        res.append(cur)

    # 35.15 s左右
    print(f"计算 [PR提交前一个月的comment数目] 用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_6.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")


# 4.7、4.8
# PR提交前的pr数目、提交前一个月内的pr数目
def tongji_project_before_pr_prs(json_file, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_name = 'apache/spark'
        project_name_all > 'apache_spark'
    '''
    job = 'project_prs'
    with open(json_file) as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr'] = data['url']
            tmp['date'] = datetime.datetime.strptime(data['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            pulls.append(tmp)
    pulls.reverse()
    datedelta = datetime.timedelta(days=31)
    res_header = ['pr_url', f'before_pr_{job}', f'before_pr_{job}_in_month']
    res = []
    le = 0
    for i, pr in tqdm(enumerate(pulls)):
        cur = []
        cur.append(pr['pr'])
        cur.append(i)
        while le < len(pulls) and pr['date'] - datedelta > pulls[le]['date']:
            le += 1
        cur.append(i - le)
        res.append(cur)
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_7-4_8.csv')
    write_csv_data(csv_file_name, res_header, res)

# 4.9、4.10
# PR提交前的issue's comment数目、提交前一个月的issue's comment数目
# issue、issue_in_month。
def tongji_project_before_pr_issues(json_file, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_name = 'apache/spark'
        project_name_all > 'apache_spark'
    '''

    job = 'project_issues_comment'
    timer = Timer()
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr'] = data['url']
            tmp['pr_id'] = data['id']
            tmp['login'] = data['user']['login']
            tmp['date'] = data['created_at']
            pulls.append(tmp)

    print(f"加载pulls用时{timer.stop()}")
    # 5s
    # res_header = ['pr_url', 'login', f'before_pr_issue_comments_{job}', f'before_pr_issue_comments_{job}_in_month']
    res_header = ['pr_url', 'login', f'before_pr_issue', f'before_pr_issue_in_month', f'before_pr_issue_comments', f'before_pr_issue_comments_in_month']
    res = []

    pulls.reverse()

    repo_name = repo_name.split("/")
    owner = repo_name[0]
    project_name = repo_name[1]
    owner_sq = f"select id from users where login = '{owner}';"
    owner_id = run_sql_mysqlclient(owner_sq)
    project_sq = f"select id from projects where owner_id = {owner_id[0]['id']} and name = '{project_name}';" # 查出owner_id
    project_id = run_sql_mysqlclient(project_sq)
    # print(project_id) # 164273219

    # issues
    issues_sq = f"select id, created_at from issues where repo_id = {project_id[0]['id']};"
    issues = list(run_sql_mysqlclient(issues_sq))
    print("issues len = ", len(issues))
    issues.sort(key=lambda x : x['created_at'])
    # comments
    comments = []
    # add 多线程操作
    cmds = []
    for item in issues:
        # print('2', item['issue_id'])
        comment_sq = f"select * from issue_comments where issue_id = {item['id']};"
        cmds.append(comment_sq)
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    for data in duo_xian_cheng_res:
        comments.extend(list(data['answer']))
    comments.sort(key=lambda x : x['created_at'])

    datedelta = datetime.timedelta(days=31)
    issue_le, issue_ri, issue_cnt = 0, 0, 0
    comment_le, comment_ri, comment_cnt = 0, 0, 0

    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        # issues cnt
        while issue_ri < len(issues) and (isinstance(issues[issue_ri]['created_at'], str) or issues[issue_ri]['created_at']==None):
            issue_ri += 1
        while issue_ri < len(issues) and pr_date > issues[issue_ri]['created_at']:
            issue_ri += 1
            issue_cnt += 1

        while issue_le < len(issues) and (isinstance(issues[issue_le]['created_at'], str) or issues[issue_le]['created_at']==None):
            issue_le += 1
        while issue_le < len(issues) and pr_date - datedelta > issues[issue_le]['created_at']:
            issue_le += 1

        cur.append(issue_cnt)
        cur.append(issue_ri - issue_le)

        # commments
        while comment_ri < len(comments) and (isinstance(comments[comment_ri]['created_at'], str) or comments[comment_ri]['created_at']==None):
            comment_ri += 1
        while comment_ri < len(comments) and pr_date > comments[comment_ri]['created_at']:
            comment_ri += 1
            comment_cnt += 1

        while comment_le < len(comments) and (isinstance(comments[comment_le]['created_at'], str) or comments[comment_le]['created_at']==None):
            comment_le += 1
        while comment_le < len(comments) and pr_date - datedelta > comments[comment_le]['created_at']:
            comment_le += 1

        cur.append(comment_cnt)
        cur.append(comment_ri - comment_le)
        res.append(cur)

    print(f"计算 [PR提交前的issue_comment数目、提交前一个月的issue_comment数目] 用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_9-4_10.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")
