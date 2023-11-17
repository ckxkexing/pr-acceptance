"""
     对项目的其他PR进行信息统计
"""
import datetime
import os

from tqdm import tqdm

from utils.common import read_pulls
from utils.files import write_csv_data
from utils.multi_thread import duo_xian_cheng
from utils.mysql_link import run_sql_mysqlclient

from .user import get_user_id


def get_project_id_query(owner_id, project_name):
    return f"select id from projects where owner_id = {owner_id} and name = '{project_name}';"


def get_project_id(owner_id, project_name):
    mq = get_project_id_query(owner_id, project_name)
    id_q = run_sql_mysqlclient(mq)
    return id_q[0]["id"] if id_q else None


def get_project_commits_created_at(project_id):
    date_sq = f"select created_at from commits where project_id = {project_id};"
    return run_sql_mysqlclient(date_sq)


def get_project_issues(project_id):
    issues_sq = f"select id, created_at from issues where repo_id = {project_id};"
    return list(run_sql_mysqlclient(issues_sq))


def get_project_pulls(project_id):
    date_sq = f"select pull_request_id from issues where repo_id = {project_id};"
    return run_sql_mysqlclient(date_sq)


def get_issue_comment_query(issue_id):
    return f"select * from issue_comments where issue_id = {issue_id};"


# PR提交前commits个数
def tongji_project_before_pr_commits(json_file, repo_name, csv_file_path):
    job = "project_commits"
    project_name_all = repo_name.replace("/", "_")
    pulls = read_pulls(json_file)

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []
    repo_name = repo_name.split("/")
    owner = repo_name[0]
    project_name = repo_name[1]

    owner_id = get_user_id(owner)
    project_id = get_project_id(owner_id, project_name)
    project = get_project_commits_created_at(project_id)
    idx, cnt = 0, 0
    pulls.reverse()

    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])

        pr_date = datetime.datetime.strptime(pr["date"], "%Y-%m-%dT%H:%M:%SZ")
        # 0000-00-00 00:00:00 => means time is undefined
        while idx < len(project) and (
            isinstance(project[idx]["created_at"], str)
            or project[idx]["created_at"] is None
        ):
            idx += 1
        while idx < len(project) and pr_date > project[idx]["created_at"]:
            cnt += 1
            idx += 1
        cur.append(cnt)
        res.append(cur)

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_3.csv"
    )
    write_csv_data(csv_file_name, res_header, res)


# pr提交前一个月commits个数
def tongji_project_before_pr_commits_in_months(
    json_file, repo_name, project_name_all, csv_file_path
):
    job = "project_commits_in_month"
    pulls = read_pulls(json_file)

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []
    repo_name = repo_name.split("/")
    owner = repo_name[0]
    project_name = repo_name[1]
    owner_id = get_user_id(owner)
    project_id = get_project_id(owner_id, project_name)
    project = get_project_commits_created_at(project_id)
    le, ri = 0, 0
    pulls.reverse()
    datedelta = datetime.timedelta(days=31)
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])

        while ri < len(project) and (
            isinstance(project[ri]["created_at"], str)
            or project[ri]["created_at"] is None
        ):
            ri += 1
        while ri < len(project) and pr["date"] > project[ri]["created_at"]:
            ri += 1

        while le < len(project) and (
            isinstance(project[le]["created_at"], str)
            or project[le]["created_at"] is None
        ):
            le += 1
        while le < len(project) and pr["date"] - datedelta > project[le]["created_at"]:
            le += 1

        cur.append(ri - le)
        res.append(cur)
    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_4.csv"
    )
    write_csv_data(csv_file_name, res_header, res)


# pr提交前的comment数目
def tongji_project_before_pr_comments(
    json_file, repo_name, project_name_all, csv_file_path
):
    job = "project_pr_comments"
    pulls = read_pulls(json_file)

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []
    repo_name = repo_name.split("/")
    owner = repo_name[0]
    project_name = repo_name[1]
    owner_id = get_user_id(owner)
    project_id = get_project_id(owner_id, project_name)
    pull_request = get_project_pulls(project_id)

    pulls.reverse()
    comments = []
    # datedelta = datetime.timedelta(days=31)
    for idx in pull_request:
        if idx["pull_request_id"] is None:
            continue
        comment_sq = f"select * from pull_request_comments where pull_request_id = {idx['pull_request_id']};"
        comment = list(run_sql_mysqlclient(comment_sq))
        comments.extend(comment)

    comments = sorted(comments, key=lambda x: x["created_at"])

    idx, cnt = 0, 0
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])

        while idx < len(comments) and (
            isinstance(comments[idx]["created_at"], str)
            or comments[idx]["created_at"] is None
        ):
            idx += 1
        while idx < len(comments) and pr["date"] > comments[idx]["created_at"]:
            idx += 1
            cnt += 1

        cur.append(cnt)
        res.append(cur)

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_5.csv"
    )
    write_csv_data(csv_file_name, res_header, res)


# PR提交前一个月的comment数目
def tongji_project_before_pr_comments_in_months(json_file, repo_name, csv_file_path):
    job = "project_comments_in_month"
    pulls = read_pulls(json_file)

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []
    project_name_all = repo_name.replace("/", "_")
    repo_name = repo_name.split("/")
    owner = repo_name[0]
    project_name = repo_name[1]
    owner_id = get_user_id(owner)
    project_id = get_project_id(owner_id, project_name)
    pull_request = get_project_pulls(project_id)

    pulls.reverse()
    comments = []
    datedelta = datetime.timedelta(days=31)

    for idx in pull_request:
        if idx["pull_request_id"] is None:
            continue
        # code review comments
        comment_sq = f"select * from pull_request_comments where pull_request_id = {idx['pull_request_id']};"
        comment = run_sql_mysqlclient(comment_sq)
        comments.extend(comment)

    comments = sorted(comments, key=lambda x: x["created_at"])

    le, ri = 0, 0
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])

        while ri < len(comments) and (
            isinstance(comments[ri]["created_at"], str)
            or comments[ri]["created_at"] is None
        ):
            ri += 1
        while ri < len(comments) and pr["date"] > comments[ri]["created_at"]:
            ri += 1

        while le < len(comments) and (
            isinstance(comments[le]["created_at"], str)
            or comments[le]["created_at"] is None
        ):
            le += 1
        while (
            le < len(comments) and pr["date"] - datedelta > comments[le]["created_at"]
        ):
            le += 1

        cur.append(ri - le)
        res.append(cur)

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_6.csv"
    )
    write_csv_data(csv_file_name, res_header, res)


def tongji_project_before_pr_prs(json_file, repo_name, csv_file_path):
    """
    "before_pr_project_prs"
    "before_pr_project_prs_in_month"
    """
    job = "project_prs"
    project_name_all = repo_name.replace("/", "_")
    pulls = read_pulls(json_file)
    pulls.reverse()
    datedelta = datetime.timedelta(days=31)
    res_header = ["pr_url", f"before_pr_{job}", f"before_pr_{job}_in_month"]
    res = []
    le = 0
    for i, pr in tqdm(enumerate(pulls)):
        cur = []
        cur.append(pr["pr"])
        cur.append(i)
        while le < len(pulls) and pr["date"] - datedelta > pulls[le]["date"]:
            le += 1
        cur.append(i - le)
        res.append(cur)
    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_7-4_8.csv"
    )
    write_csv_data(csv_file_name, res_header, res)


# PR提交前的issue's comment数目、提交前一个月的issue's comment数目
# issue、issue_in_month。


def increment_index(data_list, index, threshold, key_func):
    while index < len(data_list) and (
        isinstance(key_func(data_list[index]), str)
        or key_func(data_list[index]) is None
    ):
        index += 1

    while index < len(data_list) and threshold > key_func(data_list[index]):
        index += 1
        return index


def tongji_project_before_pr_issues(
    json_file, repo_name, project_name_all, csv_file_path
):
    job = "project_issues_comment"
    pulls = read_pulls(json_file)

    res_header = [
        "pr_url",
        "login",
        "before_pr_issue",
        "before_pr_issue_in_month",
        "before_pr_issue_comments",
        "before_pr_issue_comments_in_month",
    ]
    res = []

    pulls.reverse()

    owner, project_name = repo_name.split("/")
    owner_id = get_user_id(owner)
    project_id = get_project_id(owner_id, project_name)
    issues = get_project_issues(project_id)

    issues.sort(key=lambda x: x["created_at"])

    comments = []
    cmds = [get_issue_comment_query(item["id"]) for item in issues]
    duo_xian_cheng_res = duo_xian_cheng(cmds)

    for data in duo_xian_cheng_res:
        comments.extend(data.get("answer", []))

    comments.sort(key=lambda x: x.get("created_at", datetime.datetime.min))

    datedelta = datetime.timedelta(days=31)
    issue_le, issue_ri, issue_cnt = 0, 0, 0
    comment_le, comment_ri, comment_cnt = 0, 0, 0

    for pr in tqdm(pulls):
        cur = [pr["pr"], pr["login"]]

        # Issues count
        issue_ri = increment_index(
            issues, issue_ri, pr["date"], lambda x: x["created_at"]
        )
        issue_le = increment_index(
            issues, issue_le, pr["date"] - datedelta, lambda x: x["created_at"]
        )

        cur.append(issue_cnt)
        cur.append(issue_ri - issue_le)

        # Comments count
        comment_ri = increment_index(
            comments, comment_ri, pr["date"], lambda x: x["created_at"]
        )
        comment_le = increment_index(
            comments, comment_le, pr["date"] - datedelta, lambda x: x["created_at"]
        )

        cur.append(comment_cnt)
        cur.append(comment_ri - comment_le)
        res.append(cur)

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_9-4_10.csv"
    )
    write_csv_data(csv_file_name, res_header, res)
