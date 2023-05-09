'''
     对项目的其他PR进行信息统计
'''
import os
import json
import heapq
import pymongo
import datetime
from .utils import Timer
from .utils import write_csv_data, read_csv_data_as_dict
from tqdm import tqdm

from git import Repo

def time_handler(target_time: str):
    _date = datetime.datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
    return _date

def timestamp_handler(tim: int):
    date = datetime.datetime.fromtimestamp(tim)
    return date

# 4.3
# PR提交前commits个数
def tongji_project_before_pr_commits_in_cloned(json_file, repo_clone_dir, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_dir = '/data1/chenkexing/clone_star_50_projects_only_commits/xxx'
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

    included_commit = {}
    repo = Repo(repo_clone_dir)
    all_commits = repo.iter_commits('--all', max_count=None)
    commits_list = []
    for commit in all_commits:
        if commit.hexsha in included_commit:
            continue
        included_commit[commit.hexsha] = 1    
        commits_list.append(commit.committed_date) # 1660871609 as int.

    commits_list.sort()

    print(f"加载pulls用时{timer.stop()}")
    res_header = ['pr_url', 'login', f'before_pr_{job}']
    res = []
    idx, cnt = 0, 0
    pulls.reverse()
    
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])

        pr_date = time_handler(pr['date'])
        while idx < len(commits_list) and timestamp_handler(commits_list[idx]) < pr_date:
            idx += 1
            cnt += 1
        cur.append(cnt)
        res.append(cur)
    
    print(f"计算pulls的followers用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_3.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")

# 4.4
# PR提交1个月前commits个数
def tongji_project_before_pr_commits_in_months_in_cloned(json_file, repo_clone_dir, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_dir = '/data1/chenkexing/clone_star_50_projects_only_commits/xxx'
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
            tmp['pr']    = data['url']
            tmp['login'] = data['user']['login']
            tmp['date']  = data['created_at']
            pulls.append(tmp)
    print(f"加载pulls用时{timer.stop()}")
    
    included_commit = {}
    repo = Repo(repo_clone_dir)
    all_commits = repo.iter_commits('--all', max_count=None)
    commits_list = []
    for commit in all_commits:
        if commit.hexsha in included_commit:
            continue
        included_commit[commit.hexsha] = 1    
        commits_list.append(commit.committed_date) # 1660871609 as int.
    
    commits_list.sort()

    res_header = ['pr_url', 'login', f'before_pr_{job}']
    res = []
    le, ri = 0, 0
    pulls.reverse()
    datedelta = datetime.timedelta(days=31)
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])

        pr_date = time_handler(pr['date'])
        while ri < len(commits_list) and timestamp_handler(commits_list[ri]) < pr_date:
            ri += 1
        while le < len(commits_list) and timestamp_handler(commits_list[le]) < pr_date - datedelta:
            le += 1
        cur.append(ri - le)
        res.append(cur)
    
    print(f"计算 [ pr提交前一个月commits个数 ] 用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_4.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")

# 4.5
# pr提交前的comment数目
def tongji_project_before_pr_comments_use_mongo(json_file, issue_file, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '*apache_spark_pulls.json'
        issue_file = '/data1/chenkexing/star_50/issues/apache_echarts_repos_issues.jsonlines'
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

    pulls.reverse()
    comments = []

    # Connect to MongoDB.
    host = '127.0.0.1'
    port = 27017
    client = pymongo.MongoClient(host = host, port = port)
    db = client["issue_comments"]
    col = db[project_name_all.replace('-', '_')]

    all_data = col.find()
    mp = {}
    for info in tqdm(all_data):
        mp[info['comments_url']] = {'issue_comments': info['issue_comments']}
    
    issue_visit = {}
    pr_comments_list = []
    with open(issue_file, 'r') as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            lin_dict = json.loads(lin)
            if lin_dict['url'] in issue_visit:
                continue
            issue_visit[lin_dict['url']] = 1

            if 'pull_request' not in lin_dict:
                continue

            issue_comments_url = lin_dict['url'] + '/' + 'comments'
            comments = mp[issue_comments_url]

            for comment in comments['issue_comments']:
                pr_comments_list.append(comment['created_at'])

    pr_comments_list.sort()

    res_header = ['pr_url', 'login', f'before_pr_{job}']
    res = []

    idx, cnt = 0, 0
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        
        pr_date = time_handler(pr['date'])
        
        while idx < len(pr_comments_list) and pr_date > time_handler(pr_comments_list[idx]):
            idx += 1
            cnt += 1

        cur.append(cnt)    
        res.append(cur)
        
    print(f"计算 [pr提交前的comment数目] 用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_5.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"写入入用时{timer.stop()}")

# 4.6
# pr提交一个月前的comment数目
def tongji_project_before_pr_comments_in_months_use_mongo(json_file, issue_file, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '*apache_spark_pulls.json'
        issue_file = '/data1/chenkexing/star_50/issues/apache_echarts_repos_issues.jsonlines'
        repo_name = 'apache/spark'
        project_name_all > 'apache_spark'
    '''
    job = 'project_pr_comments_in_month'
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

    pulls.reverse()
    comments = []

    # Connect to MongoDB.
    host = '127.0.0.1'
    port = 27017
    client = pymongo.MongoClient(host = host, port = port)
    db = client["issue_comments"]
    col = db[project_name_all.replace('-', '_')]

    all_data = col.find()
    mp = {}
    for info in tqdm(all_data):
        mp[info['comments_url']] = {'issue_comments': info['issue_comments']}
    
    issue_visit = {}
    pr_comments_list = []
    with open(issue_file, 'r') as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            lin_dict = json.loads(lin)
            if lin_dict['url'] in issue_visit:
                continue
            issue_visit[lin_dict['url']] = 1

            if 'pull_request' not in lin_dict:
                continue

            issue_comments_url = lin_dict['url'] + '/' + 'comments'
            comments = mp[issue_comments_url]

            for comment in comments['issue_comments']:
                pr_comments_list.append(comment['created_at'])

    pr_comments_list.sort()

    res_header = ['pr_url', 'login', f'before_pr_{job}']
    res = []
    
    le, ri = 0, 0
    datedelta = datetime.timedelta(days=31)
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        
        pr_date = time_handler(pr['date'])
        
        while ri < len(pr_comments_list) and pr_date > time_handler(pr_comments_list[ri]):
            ri += 1

        while le < len(pr_comments_list) and pr_date - datedelta > time_handler(pr_comments_list[le]):
            le += 1
        cur.append(ri - le)    
        res.append(cur)
        
    print(f"计算 [pr提交前的comment数目] 用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_6.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"写入入用时{timer.stop()}")

# 4.9、4.10
def tongji_project_before_pr_issues_use_scrapy(json_file, issue_file, repo_name, project_name_all, csv_file_path):
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

    issues_list = []
    comments_list = []

    issue_visit = {}
    # Connect to MongoDB.
    host = '127.0.0.1'
    port = 27017
    client = pymongo.MongoClient(host = host, port = port)
    db = client["issue_comments"]
    col = db[project_name_all.replace('-', '_')]

    all_data = col.find()
    mp = {}
    for info in tqdm(all_data):
        mp[info['comments_url']] = {'issue_comments': info['issue_comments']}
    
    with open(issue_file, 'r') as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            lin_dict = json.loads(lin)
            if lin_dict['url'] in issue_visit:
                continue
            issue_visit[lin_dict['url']] = 1

            if 'pull_request' in lin_dict:
                continue
            
            issues_list.append(lin_dict['created_at'])
            issue_comments_url = lin_dict['url'] + '/' + 'comments'
            comments = mp[issue_comments_url]

            for comment in comments['issue_comments']:
                comments_list.append(comment['created_at'])
    
    issues_list.sort()
    comments_list.sort()

    res_header = ['pr_url', 'login', f'before_pr_issue', f'before_pr_issue_in_month', f'before_pr_issue_comments', f'before_pr_issue_comments_in_month']
    res = []
    pulls.reverse()
    datedelta = datetime.timedelta(days=31)
    issue_le, issue_ri, issue_cnt = 0, 0, 0
    comment_le, comment_ri, comment_cnt = 0, 0, 0

    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        pr_date = time_handler(pr['date'])
        
        while issue_ri < len(issues_list) and pr_date > time_handler(issues_list[issue_ri]):
            issue_ri += 1
            issue_cnt += 1

        while issue_le < len(issues_list) and pr_date - datedelta > time_handler(issues_list[issue_le]):
            issue_le += 1

        cur.append(issue_cnt)
        cur.append(issue_ri - issue_le)

        while comment_ri < len(comments_list) and pr_date > time_handler(comments_list[comment_ri]):
            comment_ri += 1
            comment_cnt += 1

        while comment_le < len(comments_list) and pr_date - datedelta > time_handler(comments_list[comment_le]):
            comment_le += 1

        cur.append(comment_cnt)
        cur.append(comment_ri - comment_le)
        res.append(cur)

    print(f"计算 [PR提交前的issue_comment数目、提交前一个月的issue_comment数目] 用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_{job}_4_9-4_10.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")


# 4.11 过去PR的合并比例。这里需要知道PR的close time, 需要统计出PR的merge情况。
def tongji_project_before_pr_merge_ratio(json_file, merge_info, repo_name, project_name_all, csv_file_path):
    '''
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        repo_name = 'apache/spark'
        project_name_all > 'apache_spark'
    '''

    job = 'project_merge_ratio'
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
            tmp['closed_at'] = data['closed_at']
            pulls.append(tmp)

    print(f"加载pulls用时{timer.stop()}")
    merge_info_data = read_csv_data_as_dict(merge_info)
    merge_map = {}
    for info in merge_info_data:
        merge_map[info['pr_url']] = info['merge']

    pulls.sort(key=lambda x: time_handler(x['date']))

    res = []
    res_header = ['pr_url', 'before_pr_merge_cnt', 'before_pr_closed_cnt', 'before_pr_merge_ratio']
    
    cnt , tot = 0, 0
    def cal():
        if tot == 0:
            return 0
        return cnt / tot

    class  P():
        def __init__(self, a, b):
            self.closed_at = a
            self.pr = b

        def __lt__(self, other):
            if self.closed_at == other.closed_at:
                return time_handler(self.pr['date']) < time_handler(other.pr['date'])

            if self.closed_at < other.closed_at:
                return True
            else:
                return False

    closed_queue = []

    for pr in tqdm(pulls):

        cur_time = time_handler(pr['date'])

        while closed_queue:
            next_item = heapq.heappop(closed_queue)
            if next_item.closed_at <= cur_time:
                url  = next_item.pr['pr']
                tot += 1
                if(merge_map[url] == '1'):
                    cnt += 1
            else:
                heapq.heappush(closed_queue, next_item)
                break

        cur = []
        cur.append(pr['pr'])
        cur.append(cnt)
        cur.append(tot)
        cur.append(cal())

        heapq.heappush(closed_queue, P(time_handler(pr['closed_at']), pr))
        res.append(cur)

    print(f"计算 [pr提交前的 merge PR 占比] 用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name_all}_before_pr_merge_ratio.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"写入入用时{timer.stop()}")