import datetime
import heapq
import json
import os

from git import Repo
from tqdm import tqdm

from utils.common import read_pulls
from utils.files import read_csv_data_as_dict, read_jsonline_data, write_csv_data
from utils.mongo_client import client


def time_handler(target_time: str):
    _date = datetime.datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
    return _date


def timestamp_handler(tim: int):
    date = datetime.datetime.fromtimestamp(tim)
    return date


def tongji_project_before_pr_commits_in_cloned(
    json_file, repo_clone_dir, repo_name, project_name_all, csv_file_path
):
    """
    # "before_pr_project_commits"
    """
    job = "project_commits"
    pulls = read_pulls(json_file)

    included_commit = {}
    repo = Repo(repo_clone_dir)
    all_commits = repo.iter_commits("--all", max_count=None)
    commits_list = []
    for commit in all_commits:
        if commit.hexsha in included_commit:
            continue
        included_commit[commit.hexsha] = 1
        commits_list.append(commit.committed_date)  # 1660871609 as int.

    commits_list.sort()

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []
    idx, cnt = 0, 0
    pulls.reverse()

    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])

        while (
            idx < len(commits_list)
            and timestamp_handler(commits_list[idx]) < pr["date"]
        ):
            idx += 1
            cnt += 1
        cur.append(cnt)
        res.append(cur)

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_3.csv"
    )
    write_csv_data(csv_file_name, res_header, res)


def tongji_project_before_pr_commits_in_months_in_cloned(
    json_file, repo_clone_dir, repo_name, project_name_all, csv_file_path
):
    """
    "before_pr_project_commits_in_month"
    """
    job = "project_commits_in_month"
    pulls = read_pulls(json_file)

    included_commit = {}
    repo = Repo(repo_clone_dir)
    all_commits = repo.iter_commits("--all", max_count=None)
    commits_list = []
    for commit in all_commits:
        if commit.hexsha in included_commit:
            continue
        included_commit[commit.hexsha] = 1
        commits_list.append(commit.committed_date)  # 1660871609 as int.

    commits_list.sort()

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []
    le, ri = 0, 0
    pulls.reverse()
    datedelta = datetime.timedelta(days=31)
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])

        pr_date = time_handler(pr["date"])
        while ri < len(commits_list) and timestamp_handler(commits_list[ri]) < pr_date:
            ri += 1
        while (
            le < len(commits_list)
            and timestamp_handler(commits_list[le]) < pr_date - datedelta
        ):
            le += 1
        cur.append(ri - le)
        res.append(cur)

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_4.csv"
    )
    write_csv_data(csv_file_name, res_header, res)


def tongji_project_before_pr_comments_use_mongo(
    json_file, issue_file, repo_name, project_name_all, csv_file_path
):
    """
    "before_pr_project_comments_in_pr"
    """
    job = "project_pr_comments"
    pulls = read_pulls(json_file)

    pulls.reverse()
    comments = []

    db = client["issue_comments"]
    col = db[project_name_all.replace("-", "_")]

    all_data = col.find()
    mp = {}
    for info in tqdm(all_data):
        mp[info["comments_url"]] = {"issue_comments": info["issue_comments"]}

    issue_visit = {}
    pr_comments_list = []
    with open(issue_file, "r") as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            lin_dict = json.loads(lin)
            if lin_dict["url"] in issue_visit:
                continue
            issue_visit[lin_dict["url"]] = 1

            if "pull_request" not in lin_dict:
                continue

            issue_comments_url = lin_dict["url"] + "/" + "comments"
            comments = mp[issue_comments_url]

            for comment in comments["issue_comments"]:
                pr_comments_list.append(comment["created_at"])

    pr_comments_list.sort()

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []

    idx, cnt = 0, 0
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])

        pr_date = time_handler(pr["date"])

        while idx < len(pr_comments_list) and pr_date > time_handler(
            pr_comments_list[idx]
        ):
            idx += 1
            cnt += 1

        cur.append(cnt)
        res.append(cur)

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_5.csv"
    )
    write_csv_data(csv_file_name, res_header, res)


def tongji_project_before_pr_comments_in_months_use_mongo(
    json_file, issue_file, repo_name, project_name_all, csv_file_path
):
    """
    "before_pr_project_comments_in_pr_in_month"
    """
    job = "project_pr_comments_in_month"
    pulls = read_pulls(json_file)

    pulls.reverse()
    comments = []

    db = client["issue_comments"]
    col = db[project_name_all.replace("-", "_")]

    all_data = col.find()
    mp = {}
    for info in tqdm(all_data):
        mp[info["comments_url"]] = {"issue_comments": info["issue_comments"]}

    issue_visit = {}
    pr_comments_list = []

    for lin_dict in read_jsonline_data(issue_file):
        if lin_dict["url"] in issue_visit:
            continue
        issue_visit[lin_dict["url"]] = 1

        if "pull_request" not in lin_dict:
            continue

        issue_comments_url = lin_dict["url"] + "/" + "comments"
        comments = mp[issue_comments_url]

        for comment in comments["issue_comments"]:
            pr_comments_list.append(comment["created_at"])

    pr_comments_list.sort()

    res_header = ["pr_url", "login", f"before_pr_{job}"]
    res = []

    le, ri = 0, 0
    datedelta = datetime.timedelta(days=31)
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])

        while ri < len(pr_comments_list) and pr["date"] > time_handler(
            pr_comments_list[ri]
        ):
            ri += 1

        while le < len(pr_comments_list) and pr["date"] - datedelta > time_handler(
            pr_comments_list[le]
        ):
            le += 1
        cur.append(ri - le)
        res.append(cur)

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_6.csv"
    )
    write_csv_data(csv_file_name, res_header, res)


def process_issue_join(pulls, issues_list, comments_list):
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
    datedelta = datetime.timedelta(days=31)
    issue_le, issue_ri, issue_cnt = 0, 0, 0
    comment_le, comment_ri, comment_cnt = 0, 0, 0

    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])
        pr_date = time_handler(pr["date"])

        while issue_ri < len(issues_list) and pr_date > time_handler(
            issues_list[issue_ri]
        ):
            issue_ri += 1
            issue_cnt += 1

        while issue_le < len(issues_list) and pr_date - datedelta > time_handler(
            issues_list[issue_le]
        ):
            issue_le += 1

        cur.append(issue_cnt)
        cur.append(issue_ri - issue_le)

        while comment_ri < len(comments_list) and pr_date > time_handler(
            comments_list[comment_ri]
        ):
            comment_ri += 1
            comment_cnt += 1

        while comment_le < len(comments_list) and pr_date - datedelta > time_handler(
            comments_list[comment_le]
        ):
            comment_le += 1

        cur.append(comment_cnt)
        cur.append(comment_ri - comment_le)
        res.append(cur)
    return res_header, res


def tongji_project_before_pr_issues_use_scrapy(
    json_file, issue_file, repo_name, project_name_all, csv_file_path
):
    """
    "before_pr_project_issues"
    "before_pr_project_issues_in_month"
    "before_pr_project_issues_comments"
    "before_pr_project_issues_comments_in_month"
    """
    job = "project_issues_comment"
    pulls = read_pulls(json_file)

    issues_list = []
    comments_list = []

    issue_visit = {}
    db = client["issue_comments"]
    col = db[project_name_all.replace("-", "_")]

    all_data = col.find()
    mp = {}
    for info in tqdm(all_data):
        mp[info["comments_url"]] = {"issue_comments": info["issue_comments"]}

    for lin_dict in read_jsonline_data(issue_file):
        if lin_dict["url"] in issue_visit:
            continue
        issue_visit[lin_dict["url"]] = 1

        if "pull_request" in lin_dict:
            continue

        issues_list.append(lin_dict["created_at"])
        issue_comments_url = lin_dict["url"] + "/" + "comments"
        comments = mp[issue_comments_url]

        for comment in comments["issue_comments"]:
            comments_list.append(comment["created_at"])

    issues_list.sort()
    comments_list.sort()

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}_4_9-4_10.csv"
    )
    res_header, res = process_issue_join(pulls, issues_list, comments_list)
    write_csv_data(csv_file_name, res_header, res)


def cal_rate(cnt, tot):
    if tot == 0:
        return 0
    return cnt / tot


class heap_item:
    def __init__(self, a, b):
        self.closed_at = a
        self.pr = b

    def __lt__(self, other):
        if self.closed_at == other.closed_at:
            return time_handler(self.pr["date"]) < time_handler(other.pr["date"])

        if self.closed_at < other.closed_at:
            return True
        else:
            return False


def tongji_project_before_pr_merge_ratio(
    json_file, merge_info, repo_name, project_name_all, csv_file_path
):
    """
    "before_pr_merge_cnt"
    "before_pr_closed_cnt"
    "before_pr_merge_ratio"
    """
    job = "merge_ratio"
    pulls = read_pulls(json_file)

    merge_info_data = read_csv_data_as_dict(merge_info)
    merge_map = {}
    for info in merge_info_data:
        merge_map[info["pr_url"]] = info["merge"]

    pulls.sort(key=lambda x: time_handler(x["date"]))

    res = []
    res_header = [
        "pr_url",
        "before_pr_merge_cnt",
        "before_pr_closed_cnt",
        "before_pr_merge_ratio",
    ]

    cnt, tot = 0, 0

    closed_queue = []

    for pr in tqdm(pulls):
        while closed_queue:
            next_item = heapq.heappop(closed_queue)
            if next_item.closed_at <= pr["date"]:
                url = next_item.pr["pr"]
                tot += 1
                if merge_map[url] == "1":
                    cnt += 1
            else:
                heapq.heappush(closed_queue, next_item)
                break

        cur = []
        cur.append(pr["pr"])
        cur.append(cnt)
        cur.append(tot)
        cur.append(cal_rate(cnt, tot))

        heapq.heappush(closed_queue, heap_item(pr["closed_at"], pr))
        res.append(cur)

    csv_file_name = os.path.join(
        csv_file_path, f"{project_name_all}_before_pr_{job}.csv"
    )
    write_csv_data(csv_file_name, res_header, res)
