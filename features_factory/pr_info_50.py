import os

from configs import config
from use_ghtorrent.pr_info import (
    call_pr_has_good_description,
    call_pr_tags,
    call_previous_pr,
)
from use_other.pr_cmt_message import collect_pr_commit_message, find_out_commit_type
from use_other.pr_desc import collect_pr_desc
from use_other.pr_diff import collect_pr_diff
from use_other.pr_info import (
    call_previous_issue_use_mongo,
    call_previous_merged_closed_pr,
    get_pr_comment_count,
    get_pr_commit_count,
    get_pr_commit_which_first_created,
)
from utils.files import read_json_data


def main():
    data = read_json_data(config["repo_list_file"])
    for proj in data:
        print(f"Working ON {proj['name']}")
        repo_name = proj["name"]
        project_name = proj["name"].replace("/", "_")

        json_file = f"{config['raw_data']}/pulls/{project_name}.jsonlines"
        path = f"./results/result/{project_name}"
        commit_json_file = (
            f"{config['raw_data']}/pull_commits_pre1/{project_name}_pr_commits.json"
        )
        comment_json_file = (
            f"{config['raw_data']}/pull_comment/{project_name}_pr_comments.json"
        )
        issue_json_file = (
            f"{config['raw_data']}/issues/{project_name}_repos_issues.jsonlines"
        )
        merge_info_dir = os.path.join(path, f"{project_name}_merge_info.csv")

        if not os.path.exists(path):
            os.makedirs(path)

        call_previous_pr(json_file, project_name, path)

        call_previous_issue_use_mongo(
            json_file, issue_json_file, repo_name, project_name, path
        )

        call_previous_merged_closed_pr(
            json_file, merge_info_dir, repo_name, project_name, path
        )

        call_pr_has_good_description(json_file, project_name, path)

        get_pr_commit_count(commit_json_file, project_name, path)

        get_pr_comment_count(comment_json_file, project_name, path)

        call_pr_tags(json_file, project_name, path)

        collect_pr_commit_message(commit_json_file, project_name, path)

        find_out_commit_type(commit_json_file, project_name, path)

        collect_pr_desc(json_file, project_name, path)

        collect_pr_diff(json_file, project_name, path)

        get_pr_commit_which_first_created(commit_json_file, project_name, path)


if __name__ == "__main__":
    main()
