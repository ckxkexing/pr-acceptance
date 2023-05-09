'''
chen ke xing 2021/10/22
整理从ghtorrent数据库中获取信息，所使用的代码片段。
'''
import os
import json
import os.path
import pymysql
import datetime
import numpy as np
from tqdm import tqdm
from .utils import Timer
from .utils import write_csv_data, read_csv_data_as_dict
from .test_mysql_link import get_connection, run_sql
from .test_mysql_link import get_connection_mysqlclient, run_sql_mysqlclient

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
# 统计单次PR本身相关内容
####################

# 2.1
# 这个 pulls 是否是用户在该项目的首次提交
def call_previous_pr(json_file, merge_info, project_name, csv_file_path):
    '''
        json_file = "/data1/chenkexing/star_50/human_readable_pulls/human_pull_{project_name}.json"
    '''
    job = 'check_first'
    print("### solve_for_project_pr ###")
    timer = Timer()
    user_dict = {}
    pr_flag = {}
    pr_pre = {}
    a = []
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        for data in json_data:
            uid = int(data['url'].split('/')[-1])
            data['uid'] = uid
            a.append(data)
    print(f"加载pulls用时{timer.stop()}")
    def cmpid(pr):
        return pr['uid']

    a.sort(key=cmpid)

    for i in range(len(a)):
        user = a[i]['user']['login']
        if user in user_dict:
            flag = 0
            user_dict[user] += 1
        else:
            flag = 1
            user_dict[user] = 0

        pr_flag[a[i]['url']] = flag
        pr_pre [a[i]['url']] = user_dict[user]

    res_header = ['pr_url', 'is_this_proj_first', 'number_of_pr_in_this_proj_before_pr']
    res = []

    for pr in json_data:
        cur = []
        url = pr['url']
        cur.append(url)
        cur.append(pr_flag[url])
        cur.append(pr_pre[url])
        res.append(cur)
    print(f"计算pulls的{job}个数用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_pr_{job}_2_1.csv')
    write_csv_data(csv_file, res_header, res)
    print(f"处理耗时{timer.stop()}")

# 2.1.b
def call_previous_issue(json_file, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_name = 'apache/spark'
        project_name_all = 'apache_spark'
    '''

    job = 'pr_previous_issues_in_proj'
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
    res_header = ['pr_url', 'login', \
                  f'issue_created_in_project_by_pr_author', f'issue_created_in_project_by_pr_author_in_month', \
                  f'issue_joined_in_project_by_pr_author', f'issue_joined_in_project_by_pr_author_in_month']
    res = []

    pulls.reverse()

    repo_name = repo_name.split("/")
    owner = repo_name[0]
    project_name = repo_name[1]
    owner_sq = f"select id from users where login = '{owner}';"
    owner_id = run_sql_mysqlclient(owner_sq)

    project_sq = f"select id from projects where owner_id = {owner_id[0]['id']} and name = '{project_name}';" # 查出owner_id
    project_id = run_sql_mysqlclient(project_sq)

    # issues
    issues_sq = f"select id, created_at, reporter_id from issues where repo_id = {project_id[0]['id']};"
    # issues_sq = f"select id, created_at from issues where repo_id = {project_id[0]['id']} and reporter_id={owner_id};"

    issues = list(run_sql_mysqlclient(issues_sq))
    print("issues len = ", len(issues))
    issues.sort(key=lambda x : x['created_at'])
    # comments
    comments = []
    # add 多线程操作
    cmds = []
    for item in issues:
        # user_id
        comment_sq = f"select * from issue_comments where issue_id = {item['id']};"
        cmds.append(comment_sq)

    duo_xian_cheng_res = duo_xian_cheng(cmds)
    for data in duo_xian_cheng_res:
        comments.extend(list(data['answer']))
    comments.sort(key=lambda x : x['created_at'])

    datedelta = datetime.timedelta(days=31)

    user_issue_ri, user_issue_le  = {} , {}
    user_comment_ri, user_comment_le = {} , {}
    issue_le, issue_ri, issue_cnt = 0, 0, 0
    comment_le, comment_ri, comment_cnt = 0, 0, 0

    pr_user_sqs = []
    for pr in pulls:
        pr_user_sq = f"select id from users where login = '{pr['login']}';"
        pr_user_sqs.append(pr_user_sq)
    duo_xian_cheng_res = duo_xian_cheng(pr_user_sqs)
    new_res = {}
    for data in duo_xian_cheng_res:
        new_res[data['query']] = data['answer']
    duo_xian_cheng_res = new_res

    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        pr_date = datetime.datetime.strptime(pr['date'], "%Y-%m-%dT%H:%M:%SZ")
        # pr user id
        pr_user_sq = f"select id from users where login = '{pr['login']}';"
        pr_user_id = duo_xian_cheng_res[pr_user_sq]

        if(len(pr_user_id)):
            pr_user_id = pr_user_id[0]['id']
        else:
            pr_user_id = -1
        
        # issues cnt
        while issue_ri < len(issues) and (isinstance(issues[issue_ri]['created_at'], str) or issues[issue_ri]['created_at']==None):
            issue_ri += 1

        while issue_ri < len(issues) and pr_date > issues[issue_ri]['created_at']:
            if issues[issue_ri]['reporter_id'] not in user_issue_ri:
                user_issue_ri[issues[issue_ri]['reporter_id']] = 0
            
            user_issue_ri[issues[issue_ri]['reporter_id']] += 1
            issue_ri += 1

        while issue_le < len(issues) and (isinstance(issues[issue_le]['created_at'], str) or issues[issue_le]['created_at']==None):
            issue_le += 1
        
        while issue_le < len(issues) and pr_date - datedelta > issues[issue_le]['created_at']:
            if issues[issue_le]['reporter_id'] not in user_issue_le:
                user_issue_le[issues[issue_le]['reporter_id']] = 0
            user_issue_le[issues[issue_le]['reporter_id']] += 1
            issue_le += 1
        
        if (pr_user_id!=-1 and pr_user_id in user_issue_ri):
            cur.append(user_issue_ri[pr_user_id])
            if pr_user_id not in user_issue_le:
                cur.append(user_issue_ri[pr_user_id])
            else:
                cur.append(user_issue_ri[pr_user_id] - user_issue_le[pr_user_id])
        else:
            cur.append(0)
            cur.append(0)

        # commments
        while comment_ri < len(comments) and (isinstance(comments[comment_ri]['created_at'], str) or comments[comment_ri]['created_at']==None):
            comment_ri += 1

        while comment_ri < len(comments) and pr_date > comments[comment_ri]['created_at']:
            if comments[comment_ri]['user_id'] not in user_comment_ri:
                user_comment_ri[comments[comment_ri]['user_id']] = 0
            user_comment_ri[comments[comment_ri]['user_id']] += 1
            comment_ri += 1

        while comment_le < len(comments) and (isinstance(comments[comment_le]['created_at'], str) or comments[comment_le]['created_at']==None):
            comment_le += 1
        while comment_le < len(comments) and pr_date - datedelta > comments[comment_le]['created_at']:
            if comments[comment_le]['user_id'] not in user_comment_le:
                user_comment_le[comments[comment_le]['user_id']] = 0
            user_comment_le[comments[comment_le]['user_id']] += 1            
            comment_le += 1
        if(pr_user_id != -1 and pr_user_id in user_comment_ri):
            cur.append(user_comment_ri[pr_user_id])
            if pr_user_id not in user_comment_le:
                cur.append(user_comment_ri[pr_user_id])
            else:
                cur.append(user_comment_ri[pr_user_id] - user_comment_le[pr_user_id])
        else:
            cur.append(0)
            cur.append(0)
        res.append(cur)

    print(f"计算用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_{job}_2_1_b.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")


# 2.2
def call_pr_has_good_description(json_file, project_name, csv_file_path):
    '''
        json_file = '/data1/chenkexing/merge_data/apache_spark_pulls.json'
    '''
    job = 'check_pr_description'
    
    timer = Timer()
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr'] = data['url']
            tmp['login'] = data['user']['login']
            tmp['date'] = data['created_at']
            tmp['desc'] = str(data['body'])
            pulls.append(tmp)
    
    pulls.reverse()
    length_seq = []

    res_header = ['pr_url', 'check_pr_desc_mean', 'check_pr_desc_medium','pr_desc_len']
    res = []

    for i, pr in enumerate(pulls):
        desc = str(pr['desc'])
        length = len(desc.split(' '))
        length_seq.append(length)

        mean = np.mean(length_seq)
        medium = np.median(length_seq)
        cur = []
        cur.append(pr['pr'])
        label_mean, label_medium = False, False
        if length >= mean:
            label_mean = True
        if length >= medium:
            label_medium = True
        cur.append(label_mean)
        cur.append(label_medium)
        cur.append(length)
        res.append(cur)
    print(f"计算pulls的{job}个数用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_pr_{job}_2_2.csv')
    write_csv_data(csv_file, res_header, res)
    print(f"处理耗时{timer.stop()}")


# 2.3
def call_pr_commits_count(json_file, project_name):
    '''
        实现在use_other/pr_info(自己爬的数据库中)
    '''
    pass

# 2.4
def call_pr_comments_count(json_file):
    '''
        实现在use_other/pr_info(自己爬的数据库中)
    '''
    pass


# 2.5 get pr's tags information
# this info might be created after PR for a while
def call_pr_tags(json_file, project_name, csv_file_path):
    '''
        json_file = '/data1/chenkexing/merge_data/apache_spark_pulls.json'
    '''
    job = 'tags'
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr'] = data['url']
            labels = []
            for tag in data['labels']:
                labels.append(tag['name'])
            tmp['tags'] = ','.join(labels)
            tmp['tags_count'] = len(labels)
            pulls.append(tmp)
    res_header = ['pr_url', 'tags', 'tags_count']
    res = []
    
    for i, pr in enumerate(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['tags'])
        cur.append(pr['tags_count'])
        res.append(cur)
    csv_file_name = os.path.join(csv_file_path, f'{project_name}_pr_{job}_2_5.csv')
    write_csv_data(csv_file_name, res_header, res)
