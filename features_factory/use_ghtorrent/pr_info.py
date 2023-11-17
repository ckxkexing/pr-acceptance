import datetime
import json
import os
import os.path
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from utils.common import read_pulls
from utils.files import write_csv_data
from utils.multi_thread import duo_xian_cheng
from utils.mysql_link import run_sql_mysqlclient

from .project_pr import get_issue_comment_query, get_project_id
from .user import get_user_id, get_user_id_query


def call_previous_pr(json_file, merge_info, project_name, csv_file_path):
    """
    "is_this_proj_first"
    """
    job = "check_first"
    user_dict = {}
    pr_flag = {}
    pr_pre = {}
    a = []
    with open(json_file, "r") as f:
        json_data = json.load(f)
        for data in json_data:
            uid = int(data["url"].split("/")[-1])
            data["uid"] = uid
            a.append(data)

    def cmpid(pr):
        return pr["uid"]

    a.sort(key=cmpid)

    for i in range(len(a)):
        user = a[i]["user"]["login"]
        if user in user_dict:
            flag = 0
            user_dict[user] += 1
        else:
            flag = 1
            user_dict[user] = 0

        pr_flag[a[i]["url"]] = flag
        pr_pre[a[i]["url"]] = user_dict[user]

    res_header = ["pr_url", "is_this_proj_first", "number_of_pr_in_this_proj_before_pr"]
    res = []

    for pr in json_data:
        cur = []
        url = pr["url"]
        cur.append(url)
        cur.append(pr_flag[url])
        cur.append(pr_pre[url])
        res.append(cur)
    csv_file = os.path.join(csv_file_path, f"{project_name}_pr_{job}_2_1.csv")
    write_csv_data(csv_file, res_header, res)


def call_previous_issue(json_file, repo_name, csv_file_path):
    job = "pr_previous_issues_in_proj"
    project_name_all = repo_name.replace("/", "_")
    pulls = read_pulls(json_file)

    res_header = [
        "pr_url",
        "login",
        "issue_created_in_project_by_pr_author",
        "issue_created_in_project_by_pr_author_in_month",
        "issue_joined_in_project_by_pr_author",
        "issue_joined_in_project_by_pr_author_in_month",
    ]
    res = []
    pulls.reverse()

    owner, project_name = repo_name.split("/")
    owner_id = get_user_id(owner)
    project_id = get_project_id(owner_id, project_name)

    # issues
    issues_sq = (
        f"select id, created_at, reporter_id from issues where repo_id = {project_id};"
    )
    issues = list(run_sql_mysqlclient(issues_sq))
    issues.sort(key=lambda x: x["created_at"])

    # comments
    comments = []
    cmds = [get_issue_comment_query(item["id"]) for item in issues]

    duo_xian_cheng_res = duo_xian_cheng(cmds)
    for data in duo_xian_cheng_res:
        comments.extend(data.get("answer", []))
    comments.sort(key=lambda x: x["created_at"])

    datedelta = datetime.timedelta(days=31)

    user_issue_ri, user_issue_le = defaultdict(lambda: 0), defaultdict(lambda: 0)
    user_comment_ri, user_comment_le = defaultdict(lambda: 0), defaultdict(lambda: 0)
    issue_le, issue_ri = 0, 0
    comment_le, comment_ri = 0, 0

    pr_user_sqs = [get_user_id_query(pr["login"]) for pr in pulls]
    duo_xian_cheng_res = duo_xian_cheng(pr_user_sqs)
    duo_xian_cheng_res = {data["query"]: data["answer"] for data in duo_xian_cheng_res}

    for pr in tqdm(pulls):
        cur = [pr["pr"], pr["login"]]
        # pr user id
        pr_user_sq = get_user_id_query(pr["login"])
        pr_user_id = duo_xian_cheng_res[pr_user_sq]

        if len(pr_user_id):
            pr_user_id = pr_user_id[0]["id"]
        else:
            pr_user_id = -1

        # Process issues
        issue_ri = ignore_null_created(issue_ri, issues)
        issue_ri = process_issue(issue_ri, issues, pr["date"], user_issue_ri)
        issue_le = ignore_null_created(issue_le, issues)
        issue_le = process_issue(
            issue_le, issues, pr["date"] - datedelta, user_issue_le
        )

        if pr_user_id != -1 and pr_user_id in user_issue_ri:
            cur.append(user_issue_ri[pr_user_id])
            if pr_user_id not in user_issue_le:
                cur.append(user_issue_ri[pr_user_id])
            else:
                cur.append(user_issue_ri[pr_user_id] - user_issue_le[pr_user_id])
        else:
            cur.append(0)
            cur.append(0)

        # Process comments
        comment_ri = ignore_null_created(comment_ri, comments)
        comment_ri = process_comment(comment_ri, comments, pr["date"], user_comment_ri)
        comment_le = ignore_null_created(comment_le, comments)
        comment_le = process_comment(
            comment_le, comments, pr["date"] - datedelta, user_comment_le
        )

        if pr_user_id != -1 and pr_user_id in user_comment_ri:
            cur.append(user_comment_ri[pr_user_id])
            if pr_user_id not in user_comment_le:
                cur.append(user_comment_ri[pr_user_id])
            else:
                cur.append(user_comment_ri[pr_user_id] - user_comment_le[pr_user_id])
        else:
            cur.append(0)
            cur.append(0)
        res.append(cur)

    csv_file_name = os.path.join(csv_file_path, f"{project_name_all}_{job}_2_1_b.csv")
    write_csv_data(csv_file_name, res_header, res)


def ignore_null_created(index, datas):
    while index < len(datas) and (
        isinstance(datas[index]["created_at"], str)
        or datas[index]["created_at"] is None
    ):
        index += 1
    return index


def process_issue(index, datas, date, user_issue_map):
    while index < len(datas) and date > datas[index]["created_at"]:
        user_issue_map[datas[index]["reporter_id"]] += 1
        index += 1
    return index


def process_comment(index, datas, pr_date, user_comment_map):
    while index < len(datas) and pr_date > datas[index]["created_at"]:
        user_comment_map[datas[index]["user_id"]] += 1
        index += 1
    return index


def call_pr_has_good_description(json_file, project_name, csv_file_path):
    """
    "pr_desc_len",
    "check_pr_desc_mean",
    "check_pr_desc_medium"
    """
    job = "check_pr_description"

    pulls = []
    with open(json_file, "r") as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            data = json.loads(lin)

            pulls.append(
                {
                    "pr": data["url"],
                    "login": data["user"]["login"],
                    "date": data["created_at"],
                    "desc": str(data["body"]),
                }
            )

    pulls.reverse()
    length_seq = []

    res_header = ["pr_url", "check_pr_desc_mean", "check_pr_desc_medium", "pr_desc_len"]
    res = []

    for i, pr in enumerate(pulls):
        desc = str(pr["desc"])
        length = len(desc.split(" "))
        length_seq.append(length)

        mean = np.mean(length_seq)
        medium = np.median(length_seq)
        cur = []
        cur.append(pr["pr"])
        label_mean, label_medium = False, False
        if length >= mean:
            label_mean = True
        if length >= medium:
            label_medium = True
        cur.append(label_mean)
        cur.append(label_medium)
        cur.append(length)
        res.append(cur)
    csv_file = os.path.join(csv_file_path, f"{project_name}_pr_{job}_2_2.csv")
    write_csv_data(csv_file, res_header, res)


def call_pr_commits_count(json_file, project_name):
    """
    AT use_other/pr_info
    """
    pass


def call_pr_comments_count(json_file):
    """
    AT use_other/pr_info
    """
    pass


def call_pr_tags(json_file, project_name, csv_file_path):
    """
    tags_count
    """
    job = "tags"
    pulls = []
    with open(json_file, "r") as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            data = json.loads(lin)

            tmp = {}
            tmp["pr"] = data["url"]
            labels = []
            for tag in data["labels"]:
                labels.append(tag["name"])
            tmp["tags"] = ",".join(labels)
            tmp["tags_count"] = len(labels)
            pulls.append(tmp)
    res_header = ["pr_url", "tags", "tags_count"]
    res = []

    for i, pr in enumerate(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["tags"])
        cur.append(pr["tags_count"])
        res.append(cur)
    csv_file_name = os.path.join(csv_file_path, f"{project_name}_pr_{job}_2_5.csv")
    write_csv_data(csv_file_name, res_header, res)
