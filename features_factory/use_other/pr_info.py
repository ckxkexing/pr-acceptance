import os
import json
import heapq
import pymongo
import datetime
from tqdm import tqdm

from .utils import write_csv_data, read_csv_data_as_dict
from .utils import Timer

from git import Repo
from pydriller import Repository
import networkx as nx

import vthread
import sys
sys.setrecursionlimit(10000)

def time_handler(target_time: str):
    _date = datetime.datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
    return _date

def timestamp_handler(tim: int):
    date = datetime.datetime.fromtimestamp(tim)
    return date
    
# 2.1.b use crawl github issue
def call_previous_issue_use_mongo(json_file, issue_file, repo_name, project_name, csv_file_path):
    '''
        统计 该user 之前 在项目中 创建issue和 参与issue的数目
        json_file > '/data1/chenkexing/merge_data/apache_spark_pulls.json'
        issue_file > '/data1/chenkexing/star_50/issues/apache_echarts_repos_issues.jsonlines'
        repo_name = 'apache/spark'
        project_name = 'apache_spark'
    '''
    job = 'pr_previous_issues_in_proj'
    timer = Timer()

    with open(json_file, 'r') as f:
        json_data = json.load(f)
        pulls = []
        for data in json_data:
            tmp = {}
            tmp['pr'] =    data['url']
            tmp['pr_id'] = data['id']
            tmp['login'] = data['user']['login']
            tmp['date'] =  data['created_at']
            pulls.append(tmp)
    print(f"加载pulls用时{timer.stop()}")
    
    user_issue_mp = {}
    user_issue_comment_mp = {}
    issue_visit = {}

    # Connect to MongoDB.
    host = '127.0.0.1'
    port = 27017
    client = pymongo.MongoClient(host = host, port = port)
    db = client["issue_comments"]
    col = db[project_name.replace('-', '_')]

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

            login = lin_dict['user']['login']
    
            if login not in user_issue_mp:
                user_issue_mp[login] = []
            
            user_issue_mp[login].append(lin_dict['created_at'])
            
            issue_comments_url = lin_dict['url'] + '/' + 'comments'
            comments = mp[issue_comments_url]

            for comment in comments['issue_comments']:
                login = comment['user']['login']
                if login not in user_issue_comment_mp:
                    user_issue_comment_mp[login] = []
                user_issue_comment_mp[login].append(comment['created_at'])

    res_header = ['pr_url', 'login', \
                  f'issue_created_in_project_by_pr_author', f'issue_created_in_project_by_pr_author_in_month', \
                  f'issue_joined_in_project_by_pr_author', f'issue_joined_in_project_by_pr_author_in_month']
    res = []

    pulls.reverse()
    datedelta = datetime.timedelta(days=31)
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['login'])
        pr_date = time_handler(pr['date'])
        login = pr['login']
        created_cnt, created_cnt_in_month = 0, 0

        if login in user_issue_mp:
            for tm in user_issue_mp[login]:
                tm = time_handler(tm)
                if tm <= pr_date:
                    created_cnt += 1
                    if tm + datedelta > pr_date:
                        created_cnt_in_month += 1

        joined_cnt, joined_cnt_in_month = 0, 0
        if login in user_issue_comment_mp:
            for tm in user_issue_comment_mp[login]:
                tm = time_handler(tm)
                if tm <= pr_date:
                    joined_cnt += 1
                    if tm + datedelta > pr_date:
                        joined_cnt_in_month += 1

        cur.extend([created_cnt, created_cnt_in_month, joined_cnt, joined_cnt_in_month])
        res.append(cur)
    print(f"计算用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name}_{job}_2_1_b.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")

# 2.1.c user previous merged info
def call_previous_merged_closed_pr(json_file, merge_info, repo_name, project_name, csv_file_path):
    '''
        json_file = "/data1/chenkexing/star_50/human_readable_pulls/human_pull_{project_name}.json"
        user 在project 中 merge过得、closed了的的PR个数以及占比。
    '''
    job = 'before_pr_user_merged_or_closed_pr_in_proj'
    print("### solve for pr_user_merged_or_closed_pr_in_proj###")
    timer = Timer()
    user_dict = {}
    pr_closed = {}
    pr_merged = {}
    
    a = []
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        for data in json_data:
            a.append(data)
    
    print(f"加载pulls用时{timer.stop()}")

    merge_info_data = read_csv_data_as_dict(merge_info)
    merge_map = {}

    for info in merge_info_data:
        merge_map[info['pr_url']] = info['merge']

    a.sort(key=lambda x: time_handler(x['created_at']))


    res_header = [  'pr_url',  'number_of_merged_pr_in_this_proj_before_pr', \
                               'number_of_closed_pr_in_this_proj_before_pr', \
                               'ratio_of_merged_pr_in_this_proj_before_pr' \
                 ]
    res = []
    
    def cal(user):
        if pr_closed[user] == 0:
            return 0
        return pr_merged[user] / pr_closed[user]

    class  P():
        def __init__(self, a, b):
            self.closed_at = a
            self.pr = b

        def __lt__(self, other):
            if self.closed_at == other.closed_at:
                return time_handler(self.pr['created_at']) < time_handler(other.pr['created_at'])

            if self.closed_at < other.closed_at:
                return True
            else:
                return False

    closed_queue = []

    for pr in tqdm(a):
        cur_time = time_handler(pr['created_at'])

        while closed_queue:
            next_item = heapq.heappop(closed_queue)
            if next_item.closed_at <= cur_time:
                user = next_item.pr['user']['login']
                url  = next_item.pr['url']
                pr_closed[user] += 1
                if merge_map[url] == '1':
                    pr_merged[user] += 1
            else:
                heapq.heappush(closed_queue, next_item)
                break

        cur = []
        url = pr['url']
        cur.append(url)

        user = pr['user']['login']
        if user not in user_dict:
            user_dict[user] = 1
            pr_closed[user] = 0
            pr_merged[user] = 0

        cur.append(pr_merged[user])
        cur.append(pr_closed[user])
        cur.append(cal(user))

        heapq.heappush(closed_queue, P(time_handler(pr['closed_at']), pr))
        res.append(cur)

    print(f"计算pulls的{job}个数用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_pr_{job}_2_1_c.csv')
    write_csv_data(csv_file, res_header, res)
    print(f"处理耗时{timer.stop()}")


# 2.3
def get_pr_commit_count(file_name, project_name, csv_file_path):
    '''
        file_name = {proj}_pull_commits.json
        统计PR的commit个数
    '''
    
    job = 'commit_count'
    timer = Timer()
    pulls = []
    try:
        with open(file_name, 'r') as f:
            json_data = json.load(f)
            for data in json_data:
                tmp = {}
                tmp['pr'] = data['pull_url']
                tmp['pr_commit_count'] = len(data['commits'])
                pulls.append(tmp)
    except json.decoder.JSONDecodeError:
            print("empty!")
    
    res_header = ['pr_url', 'pr_commit_count']
    res = []

    for pr in pulls:
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['pr_commit_count'])
        res.append(cur)
    print(f"计算pulls的{job}个数用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_pr_{job}_2_3.csv')
    write_csv_data(csv_file, res_header, res)
    print(f"处理耗时{timer.stop()}")


def get_pr_comment_count(file_name, project_name, csv_file_path):
    '''
        file_name = {proj}_pull_comments.json
        统计PR的comment个数
    '''
    
    job = 'comment_count'
    timer = Timer()
    pulls = []
    try:
        with open(file_name, 'r') as f:
            json_data = json.load(f)
            for data in json_data:
                tmp = {}
                tmp['pr'] = data['pull_url']
                tmp['pr_comment_count'] = len(data['comments'])
                pulls.append(tmp)
    except json.decoder.JSONDecodeError:
            print("empty!")
    
    res_header = ['pr_url', 'pr_comment_count']
    res = []

    for pr in pulls:
        cur = []
        cur.append(pr['pr'])
        cur.append(pr['pr_comment_count'])
        res.append(cur)
    print(f"计算pulls的{job}个数用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_pr_{job}_2_4.csv')
    write_csv_data(csv_file, res_header, res)
    print(f"处理耗时{timer.stop()}")


def get_pr_commit_which_first_created(file_name, project_name, csv_file_path):
    '''
        file_name = {proj}_pull_commits.json
        PR是否早于first commit进行创建。
    '''
    
    job = 'whether_pr_created_before_commit'
    timer = Timer()
    try:
        with open(file_name, 'r') as f:
            json_data = json.load(f)
    except json.decoder.JSONDecodeError:
            print("empty!")
    
    res_header = ['pr_url', f'{job}']
    res = []
    cnt = 0
    for pr in json_data:
        cur = []
        pr_tm = time_handler(pr['pr_created_time'])
        url = pr["pull_url"]
        pr_url = '/'.join(url.split('/')[:-1])
        cur.append(pr_url)
        if(len(pr['commits']) == 0):
            cm_tm = time_handler('9999-09-10T18:43:29Z')
        else:
            cm_tm = time_handler(pr['commits'][0]['commit']['author']['date'])
        if pr_tm < cm_tm:
            cur.append(True)
            cnt += 1
        else:
            cur.append(False)
        res.append(cur)
    print("pr < cmt cnt = ", cnt)
    print(f"计算pulls的{job}个数用时{timer.stop()}")
    csv_file = os.path.join(csv_file_path, f'{project_name}_pr_{job}.csv')
    write_csv_data(csv_file, res_header, res)

@vthread.pool(10)
def cal_pr_eigenvector_centrality(file_name, pr_pre_commits_dir, git_clone_commits_dir, project_name, csv_file_path):
    '''
        Calculate every PR's eigenvector: sum of every commit's (mean of file eigenvector) .
    '''

    import pytz
    utc = pytz.UTC
    
    job = 'pr_eigenvector_value'
    timer = Timer()

    # Read pulls requests info.
    pulls = []
    with open(file_name, 'r') as f:
        json_data = json.load(f)
        for data in json_data:
            pr = {}
            pr['url'] = data['url']
            pr['created_at'] = time_handler(data['created_at']).replace(tzinfo=utc)
            pr['commits_url'] = data['commits_url']
            pulls.append(pr)

    # Get PR's pre-commit from pr_pre_commits_dir.
    new_pulls = []
    with open(pr_pre_commits_dir, 'r') as f:
        json_data = json.load(f)
        pr_pre_commits_mp = {}
        for data in json_data:
            pr_pre_commits_mp[data['pull_url']] = data['commits']   # { 'xxx/commits' : [commits]}
        for i , pr in enumerate(pulls):
            tmp = []
            
            if pr['commits_url'] not in pr_pre_commits_mp:
                continue
            for commit in pr_pre_commits_mp[pr['commits_url']]:
                cur = {}
                cur['url'] = commit['url']
                cur['created_at'] = time_handler(commit['commit']['author']['date']).replace(tzinfo=utc)
                tmp.append(cur)
            pulls[i]['pre_commits'] = tmp   # { commits_url, created_at }
            new_pulls.append(pulls[i])
    pulls = new_pulls
    # Get PR's pre commit details from Mongo.
    host = '127.0.0.1'
    port = 27017
    client = pymongo.MongoClient(host = host, port = port)
    db = client["commit_details"]
    col = db[project_name.replace('-', '_')]

    all_data = col.find()
    commit_detail_mp = {}
    for info in tqdm(all_data):
        commit_detail_mp[info['commit_url']] = info['commit_detail']

    for i , pr in enumerate(pulls):
        for j , cmt in enumerate(pr['pre_commits']):
            url = cmt['url']
            if url in commit_detail_mp:
                pulls[i]['pre_commits'][j]['detail'] = commit_detail_mp[url]
            else:
                print(url)

    # Get repo's commits filename by cloned repo file.
    repo_commits = []
    for commit in tqdm(Repository(git_clone_commits_dir, include_refs=True, include_remotes=True).traverse_commits()):
        cur = {}
        cur['created_at'] = commit.author_date.replace(tzinfo=utc)  # type is datetime.datetime
        cur['commit_source_code_file'] = []
        cur['commit_nloc'] = []
        cur['commit_complexity'] = [] # Cyclomatic Complexity of the file in commits
        for m in commit.modified_files:
            if m.nloc != None and m.nloc > 0:
                if m.change_type.name == 'DELETE':
                    cur['commit_source_code_file'].append(('rm', m.old_path))
                elif m.change_type.name == 'ADD':
                    cur['commit_source_code_file'].append(('ad', m.new_path))
                elif m.change_type.name == 'MODIFY':
                    cur['commit_source_code_file'].append(('md', m.new_path))
                elif m.change_type.name == 'RENAME':
                    cur['commit_source_code_file'].append(('mv', m.old_path, m.new_path))
        repo_commits.append(cur)

    repo_commits.sort(key=lambda x: x['created_at'])
    pulls.sort(key=lambda x: x['created_at'])

    res_header = ['pr_url', f'{job}']
    res = []
    G = nx.Graph()
    commit_id = 0
    for pull in tqdm(pulls):
        cur = []
        cur.append(pull['url'])
        pr_time = pull['created_at']
        while ( commit_id < len(repo_commits) and repo_commits[commit_id]['created_at'] <= pr_time ):
            operators = repo_commits[commit_id]['commit_source_code_file']
            file_names = []
            for data in operators:
                if data[0] == 'rm':
                    G.remove_node(data[1])
                elif data[0] == 'mv':
                    # Rename the node's name.
                    mapping = {data[1]:data[2]}
                    G = nx.relabel_nodes(G, mapping)
                    # Push to queue to add edge OR update edge weight.
                    file_names.append(data[2])
                elif data[0] == 'ad' or data[0] == 'md':
                    # Push to queue to add edge OR update edge weight.
                    file_names.append(data[1])

            for i , u in enumerate(file_names):
                for j, v in enumerate(file_names):
                    if i >= j :
                        continue
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1) 
            commit_id += 1
        # Cal eigenvector in the the code source file graph before cur PR created.
        if G.nodes:
            centrality = nx.eigenvector_centrality(G, weight='weight')
        else:
            centrality = {}
        # Sum of every commits' eigenvector in PRs
        cur_eigenvector = 0
        for commit in pull['pre_commits']:
            c = []
            if 'detail' not in commit:
                continue
            for file in commit['detail']['files']:
                filename = file['filename']
                if filename in centrality:
                    c.append(centrality[filename])
            if c:
                # Mean of every files' eigenvector in commits
                cur_eigenvector += sum(c) / len(c)
        cur.append(cur_eigenvector)
        res.append(cur)

    print(f"计算用时{timer.stop()}")
    csv_file_name = os.path.join(csv_file_path, f'{project_name}_{job}.csv')
    write_csv_data(csv_file_name, res_header, res)
    print(f"重载入用时{timer.stop()}")