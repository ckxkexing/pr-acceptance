"""
    obtain information from the ghtorrent database.
"""

import os
import os.path
from collections import defaultdict

from tqdm import tqdm

from utils.common import read_pulls
from utils.files import write_csv_data
from utils.multi_thread import duo_xian_cheng
from utils.mysql_link import run_sql_mysqlclient

###################
# sql
###################


def get_followers_query(user_id):
    return f"SELECT * FROM followers WHERE user_id='{user_id}';"


def get_user_projects_query(login):
    return f"SELECT * FROM users a JOIN projects b ON a.id = b.owner_id WHERE a.login='{login}';"


def get_user_commits_query(login):
    return f"SELECT * FROM users a JOIN commits b ON a.id = b.author_id WHERE a.login='{login}';"


def get_user_issues_query(login):
    return f"select * from users a join issues b on a.id = b.reporter_id where a.login='{login}';"


def get_user_pr_query(login):
    return (
        "SELECT * "
        "FROM users a "
        "JOIN issues b ON a.id = b.reporter_id "
        f"WHERE a.login = '{login}' "
        "AND b.pull_request_id IS NOT NULL;"
    )


def get_user_id_query(login):
    return f"SELECT id FROM users WHERE login='{login}';"


def get_commit_query(user_id):
    return (
        "SELECT author_id, project_id, created_at "
        "FROM commits "
        f"WHERE author_id = '{user_id}' "
        "ORDER BY author_id"
    )


def get_user_id(login):
    mq = get_user_id_query(login)
    id_q = run_sql_mysqlclient(mq)
    return id_q[0]["id"] if id_q else None


def cal_before_pr_user_project(json_file, project_name, csv_file_path):
    """
    "before_pr_user_projects"
    """
    pulls = read_pulls(json_file)
    pulls.sort(key=lambda kv: (kv["login"], kv["date"]))
    mp = defaultdict(lambda: [])
    for pr in pulls:
        mp[pr["login"]].append(pr)
    logins = list(mp.keys())

    cmds = []
    for login in logins:
        mq = get_user_projects_query(login)
        cmds.append(mq)
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    new_res = {}
    for data in duo_xian_cheng_res:
        new_res[data["query"]] = data["answer"]
    duo_xian_cheng_res = new_res

    res_header = ["pr_url", "login", "before_pr_user_projects"]
    res = []
    for pr in tqdm(pulls):
        cur = [pr["pr"], pr["login"]]
        mq = get_user_projects_query(pr["login"])
        projects = duo_xian_cheng_res[mq]
        cnt = 0
        for project in projects:
            if project["b.created_at"] < pr["date"]:
                cnt += 1
        cur.append(cnt)
        res.append(cur)

    csv_file = os.path.join(
        csv_file_path, f"{project_name}_before_pr_user_projects_1_2.csv"
    )
    write_csv_data(csv_file, res_header, res)


# 提交pr时刻，该用户提交的commit所覆盖的项目数
def get_author_id(logins):
    authors = []
    for login in logins:
        user_id = get_user_id(login)
        if user_id:
            authors.append(user_id)
        else:
            authors.append("x")
    return authors


def process_user_commits(authors_id):
    cmds = []
    for author_id in authors_id:
        commits_mq = get_commit_query(author_id)
        cmds.append(commits_mq)

    duo_xian_cheng_res = duo_xian_cheng(cmds)
    duo_xian_cheng_res = [data["answer"] for data in duo_xian_cheng_res]

    user_commits = defaultdict(lambda: [])

    for commits in duo_xian_cheng_res:
        for commit in commits:
            if commit["author_id"] == "":
                continue
            if commit["author_id"] not in user_commits:
                user_commits[commit["author_id"]] = []
            user_commits[commit["author_id"]].append(commit)
    return user_commits


def cal_before_pr_user_commits_project(json_file, project_name, csv_file_path):
    """
    "before_pr_user_commits_proj"
    """
    job = "user_commits_proj"
    pulls = read_pulls(json_file)
    res_header = ["pr_url", "login", f"before_pr_{job}"]
    pulls.sort(key=lambda kv: (kv["login"], kv["date"]))
    mp = defaultdict(lambda: [])
    for pr in pulls:
        mp[pr["login"]].append(pr)
    logins = list(mp.keys())
    job_res = []
    authors_id = get_author_id(logins)
    user_commits = process_user_commits(authors_id)

    idx = 0
    for login in tqdm(logins):
        # res is all the users' created commits
        idx += 1
        project_mp = {}
        ind = 0

        res = [
            s
            for s in user_commits[authors_id[idx]]
            if not isinstance(s["created_at"], str) and s["created_at"] is not None
        ]

        res.sort(key=lambda x: x["created_at"])

        for pr in mp[login]:
            cur = [pr["pr"], pr["login"]]
            # 0000-00-00 00:00:00 表示时间未知,直接跳过
            while ind < len(res) and (
                isinstance(res[ind]["created_at"], str)
                or res[ind]["created_at"] is None
            ):
                ind += 1
            while ind < len(res) and res[ind]["created_at"] <= pr["date"]:
                if res[ind]["project_id"] not in project_mp:
                    project_mp[res[ind]["project_id"]] = 1
                ind += 1
            cnt = len(project_mp)
            cur.append(cnt)
            job_res.append(cur)

    csv_file = os.path.join(csv_file_path, f"{project_name}_before_pr_{job}_1_4.csv")
    write_csv_data(csv_file, res_header, job_res)


def cal_before_pr_user_commits(json_file, project_name, csv_file_path):
    """
    "before_pr_user_commits"
    """
    job = "user_commits"
    pulls = read_pulls(json_file)
    res_header = ["pr_url", "login", f"before_pr_{job}"]

    pulls.sort(key=lambda kv: (kv["login"], kv["date"]))

    mp = defaultdict(lambda: [])
    for pr in pulls:
        mp[pr["login"]].append(pr)

    logins = list(mp.keys())
    cmds = []
    for login in logins:
        mq = get_user_commits_query(login)
        cmds.append(mq)
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    duo_xian_cheng_res = {data["query"]: data["answer"] for data in duo_xian_cheng_res}

    job_res = []
    for login in tqdm(logins):
        mq = get_user_commits_query(login)
        res = duo_xian_cheng_res[mq]
        new_res = []
        for i, s in enumerate(res):
            if isinstance(s["b.created_at"], str) or s["b.created_at"] is None:
                continue
            new_res.append(s)

        new_res.sort(key=lambda x: x["b.created_at"])

        res = new_res
        cur_commit_total = 0
        ind = 0
        for pr in mp[login]:
            cur = [pr["pr"], pr["login"]]
            # 0000-00-00 00:00:00 mean None , skip
            while ind < len(res) and (
                isinstance(res[ind]["b.created_at"], str)
                or res[ind]["b.created_at"] is None
            ):
                ind += 1
            while ind < len(res) and res[ind]["b.created_at"] <= pr["date"]:
                cur_commit_total += 1
                ind += 1
            cur.append(cur_commit_total)
            job_res.append(cur)

    csv_file = os.path.join(csv_file_path, f"{project_name}_before_pr_{job}_1_5.csv")
    write_csv_data(csv_file, res_header, job_res)


def cal_before_pr_user_prs(json_file, project_name, csv_file_path):
    """
    "before pr_user_pulls"
    """
    job = "user_pulls"

    pulls = read_pulls(json_file)

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []

    cmds = []
    vis_login = {}
    for pr in pulls:
        if pr["login"] in vis_login:
            continue
        vis_login[pr["login"]] = 1
        mq = get_user_pr_query(pr["login"])
        cmds.append(mq)

    duo_xian_cheng_res = duo_xian_cheng(cmds)
    new_res = {}
    for data in duo_xian_cheng_res:
        cur = []
        for i, s in enumerate(data["answer"]):
            if isinstance(s["b.created_at"], str) or s["b.created_at"] is None:
                continue
            cur.append(s)
        new_res[data["query"]] = cur

    duo_xian_cheng_res = new_res

    for pr in tqdm(pulls):
        cur = [pr["pr"], pr["login"]]
        mq = get_user_pr_query(pr["login"])

        projects = duo_xian_cheng_res[mq]
        cnt = 0

        for project in projects:
            if (
                isinstance(project["b.created_at"], str)
                or project["b.created_at"] is None
            ):
                continue
            if project["b.created_at"] < pr["date"]:
                cnt += 1
        cur.append(cnt)
        res.append(cur)
    csv_file = os.path.join(csv_file_path, f"{project_name}_before_pr_{job}_1_6.csv")
    write_csv_data(csv_file, res_header, res)


def cal_before_pr_user_issues(json_file, project_name, csv_file_path):
    """
    "before_pr_user_issues"
    """
    job = "user_issues"
    pulls = read_pulls(json_file)

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []

    cmds = []
    vis_login = {}
    for pr in pulls:
        if pr["login"] in vis_login:
            continue
        vis_login[pr["login"]] = 1
        mq = get_user_issues_query(pr["login"])
        cmds.append(mq)
    duo_xian_cheng_res = duo_xian_cheng(cmds)
    new_res = {}
    for data in duo_xian_cheng_res:
        cur = []
        for i, s in enumerate(data["answer"]):
            if isinstance(s["b.created_at"], str) or s["b.created_at"] is None:
                continue
            cur.append(s)
        new_res[data["query"]] = cur
    duo_xian_cheng_res = new_res

    for pr in tqdm(pulls):
        cur = [pr["pr"], pr["login"]]
        mq = get_user_issues_query(pr["login"])
        projects = duo_xian_cheng_res[mq]
        cnt = 0
        for project in projects:
            if (
                isinstance(project["b.created_at"], str)
                or project["b.created_at"] is None
            ):
                continue
            if project["b.created_at"] < pr["date"]:
                cnt += 1
        cur.append(cnt)
        res.append(cur)

    csv_file = os.path.join(csv_file_path, f"{project_name}_before_pr_{job}_1_7.csv")
    write_csv_data(csv_file, res_header, res)


def cal_before_pr_user_followers(json_file, project_name, csv_file_path):
    """
    "before_pr_user_followers"
    """
    job = "user_followers"
    pulls = read_pulls(json_file)
    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []
    cmds = []
    vis_login = {}
    for pr in pulls:
        if pr["login"] in vis_login:
            continue
        vis_login[pr["login"]] = 1
        user_id = get_user_id(pr["login"])

        if user_id is not None:
            followers_query = get_followers_query(user_id)
            cmds.append(followers_query)

    duo_xian_cheng_res = duo_xian_cheng(cmds)
    duo_xian_cheng_res = {data["query"]: data["answer"] for data in duo_xian_cheng_res}

    for pr in tqdm(pulls):
        cur = [pr["pr"], pr["login"]]
        user_id = get_user_id(pr["login"])

        if user_id is None:
            continue

        followers = duo_xian_cheng_res[get_followers_query(user_id)]
        followers = [
            s
            for s in followers
            if not isinstance(s["created_at"], str) and s["created_at"] is not None
        ]

        cnt = 0
        for event in followers:
            if isinstance(event["created_at"], str) or event["created_at"] is None:
                continue
            if event["created_at"] < pr["date"]:
                cnt += 1
        cur.append(cnt)
        res.append(cur)

    csv_file = os.path.join(csv_file_path, f"{project_name}_before_pr_{job}_1_8.csv")
    write_csv_data(csv_file, res_header, res)
    # print(f"计算pulls的{job}用时{timer.stop()}")
    # print(f"重载入用时{timer.stop()}")
