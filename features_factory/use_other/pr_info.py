import datetime
import heapq
import json
import os
import sys

from tqdm import tqdm

from utils.common import read_pulls
from utils.files import read_csv_data_as_dict, read_jsonline_data, write_csv_data
from utils.mongo_client import client

sys.setrecursionlimit(10000)


def time_handler(target_time: str):
    _date = datetime.datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
    return _date


def timestamp_handler(tim: int):
    date = datetime.datetime.fromtimestamp(tim)
    return date


def process_issue_join_in_project(pulls, user_issue_mp, user_issue_comment_mp):
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
    datedelta = datetime.timedelta(days=31)
    for pr in tqdm(pulls):
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["login"])
        pr_date = time_handler(pr["date"])
        login = pr["login"]
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

    return res_header, res


def call_previous_issue_use_mongo(
    json_file, issue_file, repo_name, project_name, csv_file_path
):
    """
    "issue_created_in_project_by_pr_author",
    "issue_created_in_project_by_pr_author_in_month",
    "issue_joined_in_project_by_pr_author",
    "issue_joined_in_project_by_pr_author_in_month",
    """
    job = "pr_previous_issues_in_proj"

    pulls = read_pulls(json_file)

    user_issue_mp = {}
    user_issue_comment_mp = {}
    issue_visit = {}

    # connect to mongo client
    db = client["issue_comments"]
    col = db[project_name.replace("-", "_")]

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

        login = lin_dict["user"]["login"]

        if login not in user_issue_mp:
            user_issue_mp[login] = []

        user_issue_mp[login].append(lin_dict["created_at"])

        issue_comments_url = lin_dict["url"] + "/" + "comments"
        comments = mp[issue_comments_url]

        for comment in comments["issue_comments"]:
            login = comment["user"]["login"]
            if login not in user_issue_comment_mp:
                user_issue_comment_mp[login] = []
            user_issue_comment_mp[login].append(comment["created_at"])

    res_header, res = process_issue_join_in_project(
        pulls, user_issue_mp, user_issue_comment_mp
    )
    csv_file_name = os.path.join(csv_file_path, f"{project_name}_{job}_2_1_b.csv")
    write_csv_data(csv_file_name, res_header, res)


def cal_rate(a, b):
    if b == 0:
        return 0
    return a / b


class heap_item:
    def __init__(self, a, b):
        self.closed_at = a
        self.pr = b

    def __lt__(self, other):
        if self.closed_at == other.closed_at:
            return time_handler(self.pr["created_at"]) < time_handler(
                other.pr["created_at"]
            )

        if self.closed_at < other.closed_at:
            return True
        else:
            return False


def call_previous_merged_closed_pr(
    json_file, merge_info, repo_name, project_name, csv_file_path
):
    """
    "number_of_merged_pr_in_this_proj_before_pr"
    "number_of_closed_pr_in_this_proj_before_pr"
    "ratio_of_merged_pr_in_this_proj_before_pr"
    """
    job = "before_pr_user_merged_or_closed_pr_in_proj"

    user_dict = {}
    pr_closed = {}
    pr_merged = {}

    pulls = read_pulls(json_file)

    merge_info_data = read_csv_data_as_dict(merge_info)
    merge_map = {}

    for info in merge_info_data:
        merge_map[info["pr_url"]] = info["merge"]

    pulls.sort(key=lambda x: x["date"])

    res_header = [
        "pr_url",
        "number_of_merged_pr_in_this_proj_before_pr",
        "number_of_closed_pr_in_this_proj_before_pr",
        "ratio_of_merged_pr_in_this_proj_before_pr",
    ]
    res = []

    closed_queue = []

    for pr in tqdm(pulls):
        while closed_queue:
            next_item = heapq.heappop(closed_queue)
            if next_item.closed_at <= pr["date"]:
                user = next_item.pr["login"]
                url = next_item.pr["pr"]
                pr_closed[user] += 1
                if merge_map[url] == "1":
                    pr_merged[user] += 1
            else:
                heapq.heappush(closed_queue, next_item)
                break

        cur = []
        url = pr["pr"]
        cur.append(url)

        user = pr["login"]
        if user not in user_dict:
            user_dict[user] = 1
            pr_closed[user] = 0
            pr_merged[user] = 0

        cur.append(pr_merged[user])
        cur.append(pr_closed[user])
        cur.append(cal_rate(pr_merged[user], pr_closed[user]))

        heapq.heappush(closed_queue, heap_item(pr["closed_at"], pr))
        res.append(cur)

    csv_file = os.path.join(csv_file_path, f"{project_name}_pr_{job}_2_1_c.csv")
    write_csv_data(csv_file, res_header, res)


def get_pr_commit_count(file_name, project_name, csv_file_path):
    """
    pr_commit_count
    """

    job = "commit_count"
    pulls = []
    try:
        with open(file_name, "r") as f:
            json_data = json.load(f)
            for data in json_data:
                tmp = {}
                tmp["pr"] = data["pull_url"]
                tmp["pr_commit_count"] = len(data["commits"])
                pulls.append(tmp)
    except json.decoder.JSONDecodeError:
        print("empty!")

    res_header = ["pr_url", "pr_commit_count"]
    res = []

    for pr in pulls:
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["pr_commit_count"])
        res.append(cur)
    csv_file = os.path.join(csv_file_path, f"{project_name}_pr_{job}_2_3.csv")
    write_csv_data(csv_file, res_header, res)


def get_pr_comment_count(file_name, project_name, csv_file_path):
    """
    pr_comment_count
    """

    job = "comment_count"
    pulls = []
    try:
        with open(file_name, "r") as f:
            json_data = json.load(f)
            for data in json_data:
                tmp = {}
                tmp["pr"] = data["pull_url"]
                tmp["pr_comment_count"] = len(data["comments"])
                pulls.append(tmp)
    except json.decoder.JSONDecodeError:
        print("empty!")

    res_header = ["pr_url", "pr_comment_count"]
    res = []

    for pr in pulls:
        cur = []
        cur.append(pr["pr"])
        cur.append(pr["pr_comment_count"])
        res.append(cur)
    csv_file = os.path.join(csv_file_path, f"{project_name}_pr_{job}_2_4.csv")
    write_csv_data(csv_file, res_header, res)


def get_pr_commit_which_first_created(file_name, project_name, csv_file_path):
    """
    whether_pr_created_before_commit
    """

    job = "whether_pr_created_before_commit"
    try:
        with open(file_name, "r") as f:
            json_data = json.load(f)
    except json.decoder.JSONDecodeError:
        print("empty!")

    res_header = ["pr_url", f"{job}"]
    res = []
    cnt = 0
    for pr in json_data:
        cur = []
        pr_tm = time_handler(pr["pr_created_time"])
        url = pr["pull_url"]
        pr_url = "/".join(url.split("/")[:-1])
        cur.append(pr_url)
        if len(pr["commits"]) == 0:
            cm_tm = time_handler("9999-09-10T18:43:29Z")
        else:
            cm_tm = time_handler(pr["commits"][0]["commit"]["author"]["date"])
        if pr_tm < cm_tm:
            cur.append(True)
            cnt += 1
        else:
            cur.append(False)
        res.append(cur)
    print("pr < cmt cnt = ", cnt)
    csv_file = os.path.join(csv_file_path, f"{project_name}_pr_{job}.csv")
    write_csv_data(csv_file, res_header, res)
