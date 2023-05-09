'''
chen ke xing 2021/10/22
整理从ghtorrent数据库中获取信息，所使用的代码片段。
'''
import os
import json
import os.path
import pymysql
import datetime
from .utils import Timer
from .utils import write_csv_data
from .test_mysql_link import get_connection, run_sql
from .test_mysql_link import get_connection_mysqlclient, run_sql_mysqlclient
from tqdm import tqdm
from collections import defaultdict



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

####################
# 统计用户相关内容 
####################

# 1.2
# 提交pr时刻，该用户所创建的项目仓库个数
def cal_before_pr_user_project(json_file, project_name, csv_file_path):
    '''
        json_file : projects' pulls file dir, as:   '* pulls.json'
        csv_file_path : 结果保存的文件路径
        project_name : 仓库名称
    '''
    print("### solve_for_user_project ###")
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

    pulls.sort(key=lambda kv: (kv['login'], kv['date']))
    mp = {}
    logins = [] 
    for pr in pulls:
        if pr['login'] in mp:
            mp[pr['login']].append(pr)
        else:
            logins.append(pr['login'])
            mp[pr['login']] = [pr]
    print("login len =", len(logins))
    cmds = []
    for login in logins:
        mq = f"select * from users a join projects b on a.id=b.owner_id where a.login='{login}';"
        cmds.append(mq)
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    new_res = {}
    for data in duo_xian_cheng_res:
        new_res[data['query']] = data['answer']
    duo_xian_cheng_res = new_res

    res_header = ['pr_url', 'login', 'before_pr_user_projects']
    res = []
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        mq = f"select * from users a join projects b on a.id=b.owner_id where a.login='{pr['login']}';"
        projects = duo_xian_cheng_res[mq]
        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        cnt = 0
        for project in projects:
            if project['b.created_at'] < pr_date:
                cnt+=1
        cur.append(cnt)
        res.append(cur)
    
    # 161s
    print(f"计算pulls的projects个数用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_before_pr_user_projects_1_2.csv')
    write_csv_data(csv_file, res_header, res)
    print(f"重载入用时{timer.stop()}")


# 1.4
# 提交pr时刻，该用户提交的commit所覆盖的项目数
def cal_before_pr_user_commits_project(json_file, project_name, csv_file_path):
    '''
        json_file = '*_pulls.json'
        project_name  = 项目名称
        csv_file_path = 文件输出路径
    '''
    job = 'user_commits_proj'
    timer = Timer()
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:  
            tmp = {}
            tmp['pr'] = data['url']
            tmp['login'] = data['user']['login']
            tmp['date'] = datetime.datetime.strptime(data['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            pulls.append(tmp)

    print(f"加载pulls用时{timer.stop()}")
    res_header = ['pr_url', 'login', f'before_pr_{job}']
    pulls.sort(key=lambda kv: (kv['login'], kv['date']))
    mp = {}
    logins = []
    for pr in pulls:
        if pr['login'] in mp:
            mp[pr['login']].append(pr)
        else:
            logins.append(pr['login'])
            mp[pr['login']] = [pr]
    job_res = []
    all_author_id = []
    authors = []

    print(f"获取 user-id 信息")
    for login in tqdm(logins):
        #mq = f"select * from users a join commits b on a.id=b.author_id where a.login='{login}'"
        mq1 = f"select id from users where login='{login}' limit 1"
        users = run_sql_mysqlclient(mq1)
        for user in users[:1]:
            # all_author_id.append(f"author_id='{user['id']}'")
            authors.append(user['id'])
        if len(users) < 1:
            authors.append('x') # 未存在对应commit信息

    timer.start()
    cmds = []
    for i, author in enumerate(authors):
        commits_mq = f"select author_id, project_id, created_at from commits where author_id='{author}' order by author_id"
        cmds.append(commits_mq)
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    new_res = []
    for data in duo_xian_cheng_res:
        new_res.append(data['answer'])
    duo_xian_cheng_res = new_res
    print("获取commit信息：", timer.stop())

    user_commits = defaultdict(lambda:[])

    for commits in duo_xian_cheng_res:
        for commit in commits:
            if commit['author_id'] == '':
                continue
            if commit['author_id'] not in user_commits:
                user_commits[commit['author_id']] = []
            user_commits[commit['author_id']].append(commit)

    idx = 0
    for login in tqdm(logins):
        # res is all the users' created commits
        res = user_commits[authors[idx]]
        idx += 1

        project_mp = {}
        ind = 0
        new_res = []
        for i, s in enumerate(res):
            if isinstance(s['created_at'], str) or s['created_at']==None:
                continue
            new_res.append(s)

        new_res.sort(key=lambda x:x['created_at'])

        res = new_res
        for pr in mp[login]:
            cur = []
            cur.append(pr['pr'])
            cur.append(pr['login'])
            # 0000-00-00 00:00:00 表示时间未知,直接跳过
            while ind < len(res) and (isinstance(res[ind]['created_at'], str) or res[ind]['created_at']==None):
                ind += 1
            while ind < len(res) and res[ind]['created_at'] <= pr['date'] :
                if res[ind]['project_id'] not in project_mp:
                    project_mp[res[ind]['project_id']] = 1
                ind += 1
            cnt = len(project_mp)
            cur.append(cnt)
            job_res.append(cur)

    # 161s
    print(f"计算pulls的{job}用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_before_pr_{job}_1_4.csv')
    write_csv_data(csv_file, res_header, job_res)
    print(f"重载入用时{timer.stop()}")


# 1.5
# 提交pr时刻，用户提交过的总commit数目
def cal_before_pr_user_commits(json_file, project_name, csv_file_path):
    '''
        json_file = '*_pulls.json'
    '''
    job = 'user_commits'
    timer = Timer()
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr'] = data['url']
            tmp['login'] = data['user']['login']
            tmp['date'] = datetime.datetime.strptime(data['created_at'], "%Y-%m-%dT%H:%M:%SZ")
            pulls.append(tmp)
    print(f"加载pulls用时{timer.stop()}")
    # 5s
    res_header = ['pr_url', 'login', f'before_pr_{job}']

    pulls.sort(key=lambda kv: (kv['login'], kv['date']))

    mp = {}
    logins = []
    for pr in pulls:
        if pr['login'] in mp:
            mp[pr['login']].append(pr)
        else:
            logins.append(pr['login'])
            mp[pr['login']] = [pr]
    
    cmds = []
    for login_id in logins:
        mq = f"select * from users a join commits b on a.id=b.author_id where a.login='{login_id}';"
        cmds.append(mq)
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    new_res = {}
    for data in duo_xian_cheng_res:
        new_res[data['query']] = data['answer']
    duo_xian_cheng_res = new_res

    job_res = []
    for login_id in tqdm(logins):
        
        mq = f"select * from users a join commits b on a.id=b.author_id where a.login='{login_id}';"
        res = duo_xian_cheng_res[mq]
        new_res = []
        for i, s in enumerate(res):
            if isinstance(s['b.created_at'], str) or s['b.created_at']==None:
                continue
            new_res.append(s)

        new_res.sort(key=lambda x:x['b.created_at'])

        res = new_res
        cur_commit_total = 0
        ind = 0
        for pr in mp[login_id]:
            cur = []
            cur.append(pr['pr'])
            cur.append(pr['login'])
            # 0000-00-00 00:00:00 表示时间未知,直接跳过
            while ind < len(res) and (isinstance(res[ind]['b.created_at'], str) or res[ind]['b.created_at']==None):
                ind += 1
            while  ind < len(res) and res[ind]['b.created_at'] <= pr['date'] :
                cur_commit_total += 1
                ind += 1
            cur.append(cur_commit_total)
            job_res.append(cur)

    print(f"计算pulls的{job}用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_before_pr_{job}_1_5.csv')
    write_csv_data(csv_file, res_header, job_res)
    print(f"重载入用时{timer.stop()}")

# 1.6
# 提交pr时刻，该用户提交的各个项目总pr数目
# check#1 mysqlclient 
# check#2 csv_file_path
# check#3 mysql 返回的None Value要跳过，比如 project['created_at']
def cal_before_pr_user_prs(json_file, project_name, csv_file_path):
    '''
        json_file = '/data1/chenkexing/merge_data/apache_spark_pulls.json'
    '''
    job = 'user_pulls'

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

    cmds = []
    vis_login = {}
    for pr in pulls:
        if pr['login'] in vis_login:
            continue
        vis_login[pr['login']] = 1
        mq = f"select * from users a join issues b on a.id=b.reporter_id where a.login='{pr['login']}' and b.pull_request_id is not null;"
        cmds.append(mq)
    
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    new_res = {}
    for data in duo_xian_cheng_res:
        cur = []
        for i, s in enumerate(data['answer']):
            if isinstance(s['b.created_at'], str) or s['b.created_at']==None:
                continue
            cur.append(s)
        new_res[data['query']] = cur

    duo_xian_cheng_res = new_res

    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        mq = f"select * from users a join issues b on a.id=b.reporter_id where a.login='{pr['login']}' and b.pull_request_id is not null;"
        projects = duo_xian_cheng_res[mq]
        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        cnt = 0

        for project in projects:
            if (isinstance(project['b.created_at'], str) or project['b.created_at']==None):
                continue
            if project['b.created_at'] < pr_date:
                cnt+=1  
        cur.append(cnt)
        res.append(cur)
    #issues 987s  pulls 100s
    print(f"### 计算pulls前{job}用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_before_pr_{job}_1_6.csv')
    write_csv_data(csv_file, res_header, res)

# 1.7
# 提交pr时刻，该用户提交的各个项目总issues数目
# check#1 _mysqlclient
# check#2 csv_file_path + os.path
#        csv_file = os.path.join(csv_file_path, f'{project_name}_before_pr_{job}_1_6.csv')
# check#3 mysql 返回的None Value要跳过，比如 project['created_at']
#        if (isinstance(project['created_at'], str) or project['created_at']==None)
def cal_before_pr_user_issues(json_file, project_name, csv_file_path):
    '''
        json_file = '/data1/chenkexing/merge_data/apache_spark_pulls.json'
    '''
    job = 'user_issues'

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

    cmds = []
    vis_login = {}
    for pr in pulls:
        if pr['login'] in vis_login:
            continue
        vis_login[pr['login']] = 1
        mq = f"select * from users a join issues b on a.id=b.reporter_id where a.login='{pr['login']}' ;"
        cmds.append(mq)
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    new_res = {}
    for data in duo_xian_cheng_res:
        cur = []
        for i, s in enumerate(data['answer']):
            if isinstance(s['b.created_at'], str) or s['b.created_at']==None:
                continue
            cur.append(s)
        new_res[data['query']] = cur
    duo_xian_cheng_res = new_res
    
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        mq = f"select * from users a join issues b on a.id=b.reporter_id where a.login='{pr['login']}' ;"
        projects = duo_xian_cheng_res[mq]
        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        cnt = 0
        for project in projects:
            if (isinstance(project['b.created_at'], str) or project['b.created_at']==None):
                continue
            if project['b.created_at'] < pr_date:
                cnt+=1
        cur.append(cnt)
        res.append(cur)
        # print(pr['pr'])
    #issues 987s  pulls 100s
    print(f"计算pulls前{job}用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_before_pr_{job}_1_7.csv')
    write_csv_data(csv_file, res_header, res)
    print(f"重载入用时{timer.stop()}")

# 1.8
# 提交pr时刻，该用户的粉丝数目
# check
# 1 mysqlclient
# 2 csv_file_path + os.path
# 3 跳过mysql返回的None值
def cal_before_pr_user_followers(json_file, project_name, csv_file_path):
    '''
        json_file = *_pulls.json
    '''
    job = 'user_followers'
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

    cmds = []
    vis_login = {}
    for pr in pulls:
        if pr['login'] in vis_login:
            continue
        vis_login[pr['login']] = 1
        mq = f"select id from users where login='{pr['login']}';"
        id_q = run_sql_mysqlclient(mq)
        if id_q:
            id = id_q[0]['id']
        else :
            continue
        mq = f"select * from followers where user_id='{id}';"
        cmds.append(mq)
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    new_res = {}
    for data in duo_xian_cheng_res:
        new_res[data['query']] = data['answer']
    duo_xian_cheng_res = new_res

    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        mq = f"select id from users where login='{pr['login']}';"
        id_q = run_sql_mysqlclient(mq)
        if id_q:
            id = id_q[0]['id']
        else :
            continue
        
        followers = duo_xian_cheng_res[f"select * from followers where user_id='{id}';"]
        
        new_res = []
        for i, s in enumerate(followers):
            if isinstance(s['created_at'], str) or s['created_at']==None:
                continue
            new_res.append(s)
        
        followers = new_res
        
        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        cnt = 0
        for event in followers:
            if (isinstance(event['created_at'], str) or event['created_at']==None):
                continue
            if event['created_at'] < pr_date:
                cnt+=1
        cur.append(cnt)
        res.append(cur)

    print(f"计算pulls的{job}用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_before_pr_{job}_1_8.csv')
    write_csv_data(csv_file, res_header, res)
    print(f"重载入用时{timer.stop()}")

