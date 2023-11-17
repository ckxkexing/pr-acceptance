import os

from configs import config
from use_ghtorrent.user import (
    cal_before_pr_user_commits,
    cal_before_pr_user_commits_project,
    cal_before_pr_user_followers,
    cal_before_pr_user_issues,
    cal_before_pr_user_project,
    cal_before_pr_user_prs,
)
from use_other.user_type import bot_detect_by_login
from utils.files import read_json_data


def main():
    data = read_json_data(config["repo_list_file"])

    for proj in data:
        print(f"Working ON {proj['name']}")

        project_name = proj["name"].replace("/", "_")
        json_file = f"{config['raw_data']}/pulls/{project_name}.jsonlines"
        path = f"results/result/{project_name}"
        if not os.path.exists(path):
            os.makedirs(path)

        cal_before_pr_user_project(json_file, project_name, path)

        cal_before_pr_user_commits_project(json_file, project_name, path)

        cal_before_pr_user_commits(json_file, project_name, path)

        cal_before_pr_user_prs(json_file, project_name, path)

        cal_before_pr_user_issues(json_file, project_name, path)

        cal_before_pr_user_followers(json_file, project_name, path)

        bot_detect_by_login(json_file, project_name, path)


if __name__ == "__main__":
    main()
