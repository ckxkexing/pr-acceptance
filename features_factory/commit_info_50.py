from configs import config
from use_ghtorrent.commit_code import tongji_project_commit_total_change
from utils.files import mkdir, read_json_data


def main():
    data = read_json_data(config["repo_list_file"])
    for proj in data:
        print(f"Working on{proj['name']}")
        project_name = proj["name"].replace("/", "_")
        path = f"results/result/{project_name}"
        mkdir(path)

        commits_json_file = (
            f"{config['raw_data']}/pull_commits_pre1/{project_name}_pr_commits.json"
        )

        tongji_project_commit_total_change(commits_json_file, project_name, path)


if __name__ == "__main__":
    main()
