import os

from configs import config
from use_ghtorrent.project_pr import tongji_project_before_pr_prs
from use_other.project_pr import (
    tongji_project_before_pr_comments_in_months_use_mongo,
    tongji_project_before_pr_comments_use_mongo,
    tongji_project_before_pr_commits_in_cloned,
    tongji_project_before_pr_commits_in_months_in_cloned,
    tongji_project_before_pr_issues_use_scrapy,
    tongji_project_before_pr_merge_ratio,
)
from utils.files import read_json_data


def main():
    data = read_json_data(config["repo_list_file"])
    total = len(data)
    cnt = 0
    for proj in data:
        cnt += 1

        print(f"[{cnt} / {total} ]Working ON{proj['name']}")
        repo_name = proj["name"]
        project_name = proj["name"].replace("/", "_")

        path = f"results/result/{project_name}"
        if not os.path.exists(path):
            os.makedirs(path)

        pulls_json_file = f"{config['raw_data']}/pulls/{project_name}.jsonlines"
        issue_json_file = (
            f"{config['raw_data']}/issues/{project_name}_repos_issues.jsonlines"
        )
        clone_dir = f"{config['raw_data']}/clone_star_50_projects_only_commits"
        repo_clone_dir = os.path.join(clone_dir, repo_name) + "/.git"
        merge_info_dir = os.path.join(path, f"{project_name}_merge_info.csv")

        tongji_project_before_pr_commits_in_cloned(
            pulls_json_file, repo_clone_dir, repo_name, project_name, path
        )

        tongji_project_before_pr_commits_in_months_in_cloned(
            pulls_json_file, repo_clone_dir, repo_name, project_name, path
        )

        tongji_project_before_pr_comments_use_mongo(
            pulls_json_file, issue_json_file, repo_name, project_name, path
        )

        tongji_project_before_pr_comments_in_months_use_mongo(
            pulls_json_file, issue_json_file, repo_name, project_name, path
        )

        tongji_project_before_pr_prs(pulls_json_file, repo_name, path)

        tongji_project_before_pr_issues_use_scrapy(
            pulls_json_file, issue_json_file, repo_name, project_name, path
        )

        tongji_project_before_pr_merge_ratio(
            pulls_json_file, merge_info_dir, repo_name, project_name, path
        )


if __name__ == "__main__":
    main()
