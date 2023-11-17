"""

     Check if information is merged

"""

import json
import os
import re

from git import Repo

from configs import config
from utils.files import read_json_data, write_csv_data


def is_sha1(maybe_sha):
    if len(maybe_sha) not in [7, 40, 12]:
        return False
    try:
        _ = int(maybe_sha, 16)
    except ValueError:
        return False
    return True


def has_sha1_in_sentence(text):
    pattern = re.compile(r"\b[0-9a-f]{7,40}\b")
    result = pattern.findall(text)
    result = "".join(result)
    if len(result) > 0:
        return True
    else:
        return False


def has_pull_in_sentence(text):
    pattern = re.compile(r"\b\#(\d+)\b")  # Match '#2333' pattern
    result = pattern.findall(text)
    result = "".join(result)
    if len(result) > 0:
        return True

    pattern_url = re.compile(
        r"\bhttps://github.com(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\b"
    )
    result = pattern_url.findall(text)
    result = "".join(result)
    if len(result) > 0:
        return True
    return False


def check_merge_from_comment(text, login, author_association="NONE"):
    # check login is or not a bot
    patterns = [r"\Wbot\W$", r"bot$", r"\Wrobot$"]
    for pattern in patterns:
        if re.findall(pattern, login):
            return 0

    text = text.lower()
    flag, ff = 0, 0
    text = text.replace(".", " ")
    text = " ".join(text.splitlines())
    text = text.split(" ")
    for i, word in enumerate(text):
        if word in ["but", "cannot", "donot", "can't", "don't", "closing", "before"]:
            break
        if word in ["merged", "landed", "pushed", "included", "committed"]:
            ff = 1
        if ff:
            next_sentence = " ".join(text[i:])
            if has_sha1_in_sentence(next_sentence):
                flag = 1
            if has_pull_in_sentence(next_sentence):
                flag = 1
            # if( author_association == 'CONTRIBUTOR'):
            #     flag = 1
            break
    # pattern = r'\Wlgtm\W'
    # text = ' '.join(text)
    # if len(text) < 10 and re.findall(pattern, text):
    #     flag = 1
    return flag


def get_repo_commit_mention_pull(msg):
    res = []
    msg = msg.splitlines()

    for text in msg:
        # 查找 start with gh- pr id
        pattern = re.compile(r"\bgh-(\d+)\b")  # type: " gh-233 "
        result = pattern.findall(text)
        res.extend(result)
        # 查找 start with # pr id
        pattern2 = re.compile(r"\b\#(\d+)\b")  # type " #2333 "
        result = pattern2.findall(text)
        # 查找 repo-commit 中pr url
        pattern_url = re.compile(
            r"\bhttps://github.com(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\b"
        )
        result = pattern_url.findall(text)
        for url in result:
            if "pull" in url and url.split("/")[-1].isnumeric():
                res.append(url.split("/")[-1])
    return res


def process_github_merge_map(github_merge_map, pulls_data):
    github_cnt, total_pr = 0, len(pulls_data)
    for pulls in pulls_data:
        if pulls["merged_at"] is not None:
            id1 = pulls["url"].split("/")
            id1 = id1[id1.index("pulls") + 1]
            github_merge_map[id1] = 1
            github_cnt += 1
    return github_cnt, total_pr


def process_comment_merge_map(comment_merge_map, project_comments_file):
    comment_cnt = 0
    with open(project_comments_file, "r") as f:
        comments_data = json.load(f)

    # judge pr merge by pulls comments
    for data in comments_data:
        flag = 0
        for d in data["comments"][-3:]:
            if check_merge_from_comment(
                d["body"], d["user"]["login"], d["author_association"]
            ):
                flag = 1
        comment_cnt += flag
        if flag:
            id2 = data["pull_url"].split("/")
            id2 = id2[id2.index("issues") + 1]
            comment_merge_map[id2] = 1
    return comment_cnt


def process_commit_mention_map(commit_mention_map, repo_clone_dir, repo_commits_fille):
    with open(repo_commits_fille, "r") as f:
        repo_commits_data = json.load(f)

    for data in repo_commits_data:
        msg = data["commit"]["message"]
        ids = get_repo_commit_mention_pull(msg)
        for id in ids:
            commit_mention_map[id] = 1

    repo = Repo(repo_clone_dir)
    all_commits = repo.iter_commits("--all", max_count=None)
    for commit in all_commits:
        msg = commit.message
        ids = get_repo_commit_mention_pull(msg)
        for id in ids:
            commit_mention_map[id] = 1


def process_commit_included_map(commit_included_map, repo_clone_dir, pr_commits_dir):
    repo = Repo(repo_clone_dir)
    all_commits = repo.iter_commits("--all", max_count=None)
    included_commit = {}
    for commit in all_commits:
        included_commit[commit.hexsha] = 1

    with open(pr_commits_dir, "r") as f:
        pr_commits_data = json.load(f)
    for pr in pr_commits_data:
        id1 = pr["pull_url"].split("/")
        id1 = id1[id1.index("pulls") + 1]
        flag = 0
        for commit in pr["commits"]:
            if commit["sha"] in included_commit:
                flag += 1
        if flag and flag == len(pr["commits"]):
            commit_included_map[id1] = 1


def cal_pr_merge_info(project_name):
    pulls_path = f"{config['raw_data']}/human_readable_pulls/"
    project_pulls_file = os.path.join(pulls_path, f"human_pull_{project_name}.json")
    with open(project_pulls_file, "r") as f:
        pulls_data = json.load(f)

    github_merge_map = {}
    github_cnt, total_pr = process_github_merge_map(github_merge_map, pulls_data)

    print("PRs", total_pr)
    print("github_merge_at:", github_cnt)

    comments_path = f"{config['raw_data']}/pull_comment"
    project_comments_file = os.path.join(
        comments_path, f"{project_name}_pr_comments.json"
    )
    comment_merge_map = {}
    comment_cnt = process_comment_merge_map(comment_merge_map, project_comments_file)
    print("comment_merge_at:", comment_cnt)

    clone_dir = f"{config['raw_data']}/clone_star_50_projects_only_commits"
    repo_clone_dir = os.path.join(clone_dir, project_name.replace("_", "/")) + "/.git"
    commit_path = f"{config['raw_data']}/repo_commits"
    repo_commits_fille = os.path.join(commit_path, f"{project_name}_repo_commits.json")
    commit_mention_map = {}
    process_commit_mention_map(commit_mention_map, repo_commits_fille, repo_clone_dir)

    commit_included_map = {}
    pr_commits_file = (
        f"{config['raw_data']}/pull_commits3/{project_name}_pr_commits.json"
    )
    process_commit_included_map(commit_included_map, repo_clone_dir, pr_commits_file)

    header = ["pr_url", "merge", "github_merge", "comment_merge", "commit_merge"]
    res = []
    total_merge = 0

    for pulls in pulls_data:
        cur = []
        cur.append(pulls["url"])
        ix = pulls["url"].split("/")
        ix = ix[ix.index("pulls") + 1]

        github_merge = 0
        if ix in github_merge_map:
            github_merge = 1
        comment_merge = 0
        if ix in comment_merge_map:
            comment_merge = 1

        commit_merge = 0
        if ix in commit_mention_map or ix in commit_included_map:
            commit_merge = 1

        merge = 0
        if github_merge + comment_merge + commit_merge:
            total_merge += 1
            merge = 1
        cur.extend([merge, github_merge, comment_merge, commit_merge])
        res.append(cur)
    print("commit_mention_cnt: ", commit_mention_cnt)
    print("commit_include_cnt: ", commit_included_cnt)
    print("total_merge:", total_merge)
    output_path = f"./results/result/{project_name}"
    output_file = os.path.join(output_path, f"{project_name}_merge_info.csv")
    write_csv_data(output_file, header, res)
    # logger.info(f"{project_name}\t{github_cnt}\t{comment_cnt}\t{commit_mention_cnt}")
    return (
        github_cnt,
        comment_cnt,
        commit_mention_cnt,
        commit_included_cnt,
        total_pr,
        total_merge,
    )


if __name__ == "__main__":
    data = read_json_data(config["repo_list_file"])
    info_header = [
        "project_name",
        "github_cnt",
        "comment_cnt",
        "commit_mention_cnt",
        "commit_included_cnt",
        "total_pr",
        "total_merge",
    ]
    res = []
    for proj in data:
        print(f"Working On {proj['name']}")

        project_name = proj["name"].replace("/", "_")

        (
            github_cnt,
            comment_cnt,
            commit_mention_cnt,
            commit_included_cnt,
            total_pr,
            total_merge,
        ) = cal_pr_merge_info(project_name)
        res.append(
            [
                project_name,
                github_cnt,
                comment_cnt,
                commit_mention_cnt,
                commit_included_cnt,
                total_pr,
                total_merge,
            ]
        )

    # write_csv_data('50_proj_merge_info.csv', info_header, res)
