"""
    remove PR's commits that created after the PR created
"""

import json
import os
import sys
from datetime import datetime, timedelta

import vthread
from tqdm import tqdm

from utils.files import mkdir, read_json_data

sys.path.append(os.path.dirname(sys.path[0]))

pulls_data_file = "/data1/chenkexing/star_50/human_readable_pulls"
commits_data_file = "/data1/chenkexing/star_50/pull_commits3"
new_commits_file = "/data1/chenkexing/star_50/pull_commits_pre1"


def time_handler(target_time: str):
    _date = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
    return _date
    local_time = _date + timedelta(hours=8)
    end_time = local_time.strftime("%Y-%m-%d %H:%M:%S")
    return end_time


@vthread.pool(6)
def solve(pulls_file, commit_file, project_name):
    pulls_data = json.load(open(pulls_file, "r"))
    commits_data = json.load(open(commit_file, "r"))
    mp = {}
    mp_origin = {}
    for info in pulls_data:
        pulls = info["url"]
        tm = time_handler(info["created_at"])
        mp[pulls] = tm
        mp_origin[pulls] = info["created_at"]

    new_commits_data = []
    for pr in tqdm(commits_data):
        new_pr = pr.copy()
        url = pr["pull_url"]
        pr_url = "/".join(url.split("/")[:-1])
        new_commits = []
        pr_time = mp[pr_url]
        flag = 0  # 保证有一个commits
        for cm in pr["commits"]:
            if time_handler(cm["commit"]["author"]["date"]) <= pr_time or flag == 0:
                flag = 1
                new_commits.append(cm)
        new_pr["commits"] = new_commits
        new_pr["pr_created_time"] = mp_origin[pr_url]
        new_commits_data.append(new_pr)
    mkdir(new_commits_file)
    new_commits_dir = os.path.join(new_commits_file, f"{project_name}_pr_commits.json")
    with open(new_commits_dir, "w") as f:
        json.dump(new_commits_data, f, indent=2)


if __name__ == "__main__":
    data = read_json_data("../star_proj_get_2021_9_7_pre50.json")
    for proj in data:
        project_name = proj["name"].replace("/", "_")

        pulls_file = os.path.join(pulls_data_file, f"human_pull_{project_name}.json")
        commits_file = os.path.join(
            commits_data_file, f"{project_name}_pr_commits.json"
        )

        solve(pulls_file, commits_file, project_name)
