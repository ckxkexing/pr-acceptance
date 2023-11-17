"""
    Get message of commit in PRs.
"""

import os

from tqdm import tqdm

from utils.files import read_json_data, write_csv_data
from utils.mongo_client import client


def collect_pr_commit_message(pulls_commit_file_dir, project_name, csv_file_path):
    """
    commit_message
    """
    pr_commit_data = read_json_data(pulls_commit_file_dir)

    header = ["pr_url", "commit_message"]
    res = []

    for pr in tqdm(pr_commit_data):
        cur = []
        # in commit file, pull_url unexpecded include text:'commit'
        tmp = pr["pull_url"].split("/")
        if "commits" in tmp:
            tmp.pop()
        cur.append("/".join(tmp))
        message = []
        for commit in pr["commits"]:
            tmps = commit["commit"]["message"].splitlines()
            tmp = "<nl>".join(tmps)
            message.append(tmp)
        cur.append(";".join(message))
        res.append(cur)

    output_file = os.path.join(csv_file_path, f"{project_name}_pr_commit_message.csv")
    write_csv_data(output_file, header, res)


def collect_pr_commit_diff(pulls_commit_file_dir, project_name, csv_file_path):
    pass


def find_out_commit_type(pulls_commit_file_dir, project_name, csv_file_path):
    """
    "contain_test_file"
    "contain_doc_file"
    """
    pr_commit_data = read_json_data(pulls_commit_file_dir)

    db = client["commit_details"]
    col = db[project_name.replace("-", "_")]
    all_data = col.find()
    mp = {}
    for info in tqdm(all_data):
        mp[info["commit_url"].split("/")[-1]] = {
            "files": info["commit_detail"]["files"]
        }

    header = ["pr_url", "contain_test_file", "contain_doc_file"]
    res = []

    for pr in tqdm(pr_commit_data):
        cur = []
        # in commit file, pull_url unexpecded include text:'commit'
        tmp = pr["pull_url"].split("/")
        if "commits" in tmp:
            tmp.pop()
        cur.append("/".join(tmp))

        test_cnt, doc_cnt = 0, 0
        for commit in pr["commits"]:
            url = commit["url"]
            # Use sha-id rather than url would be more robust.
            files = mp[url.split("/")[-1]]["files"]
            for file in files:
                filename = file["filename"]
                f = filename.split("/")
                if "test" in f[0]:
                    test_cnt += 1
                if f[0] == "doc" or f[0] == "docs":
                    doc_cnt += 1
        cur.append(test_cnt)
        cur.append(doc_cnt)
        res.append(cur)

    csv_file = os.path.join(
        csv_file_path, f"{project_name}_commit_contain_test_or_doc.csv"
    )
    write_csv_data(csv_file, header, res)
