import json
import os

import vthread
from tqdm import tqdm

from utils.files import write_csv_data
from utils.mongo_client import client


@vthread.pool(5)
def tongji_project_commit_total_change(json_file, project_name, csv_file_path):
    """
    "commit_add_line"
    "commit_delete_line"
    "commit_total_line"
    "commit_add_line_max"
    "commit_delete_line_max"
    "commit_total_line_max"
    "commit_add_line_min"
    "commit_delete_line_min"
    "commit_total_line_min"
    "commit_file_change"
    """
    job = "code_change_info"

    with open(json_file, "r") as f:
        json_data = json.load(f)

    # connect to mongo
    db = client["commit_details"]
    col = db[project_name.replace("-", "_")]
    all_data = col.find()
    mp = {}
    for info in tqdm(all_data):
        mp[info["commit_url"]] = {
            "stats": info["commit_detail"]["stats"],
            "files": info["commit_detail"]["files"],
        }

    res = []
    res_header = [
        "pr_url",
        "commit_add_line",
        "commit_delete_line",
        "commit_total_line",
        "commit_file_change",
        "commit_add_line_max",
        "commit_delete_line_max",
        "commit_total_line_max",
        "commit_add_line_min",
        "commit_delete_line_min",
        "commit_total_line_min",
    ]
    low = 0
    for pr_data in tqdm(json_data):
        cur = []
        pr_id = pr_data["pull_url"]
        id = "/".join(pr_id.split("/")[0:-1])
        cur.append(id)
        commits = pr_data["commits"]
        add, rid, tot = 0, 0, 0
        add_max, rid_max, tot_max = 0, 0, 0
        add_min, rid_min, tot_min = 1e9, 1e9, 1e9
        repo_name = project_name.split("_")
        # file_cnt = pr_data['changed_files']
        change_file = {}
        for commit in commits:
            cur_commit_url = (
                f"https://api.github.com/repos/{repo_name[0]}/{repo_name[1]}/commits/"
                + commit["sha"]
            )
            if cur_commit_url in mp:
                info = mp[cur_commit_url]
                add += info["stats"]["additions"]
                rid += info["stats"]["deletions"]
                tot += info["stats"]["total"]
                add_max = max(add_max, info["stats"]["additions"])
                rid_max = max(rid_max, info["stats"]["deletions"])
                tot_max = max(tot_max, info["stats"]["total"])

                add_min = min(add_min, info["stats"]["additions"])
                rid_min = min(rid_min, info["stats"]["deletions"])
                tot_min = min(tot_min, info["stats"]["total"])
                for file in info["files"]:
                    if file["filename"] not in change_file:
                        change_file[file["filename"]] = 1
            else:
                low += 1
        if len(commits) == 0:
            continue
        file_cnt = len(change_file)
        cur.extend(
            [
                add,
                rid,
                tot,
                file_cnt,
                add_max,
                rid_max,
                tot_max,
                add_min,
                rid_min,
                tot_min,
            ]
        )
        res.append(cur)
    csv_file = os.path.join(csv_file_path, f"{project_name}_commit_{job}_3_3-3_5.csv")
    write_csv_data(csv_file, res_header, res)
