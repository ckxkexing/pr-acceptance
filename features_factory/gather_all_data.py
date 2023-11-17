import csv
import os
import sys

import pandas as pd

from configs import config
from utils.files import (
    mkdir,
    read_csv_data_as_dict,
    read_json_data,
    write_csv_data_as_list_of_dict,
)

csv.field_size_limit(sys.maxsize)


def read_data(project_name):
    pr_input = pd.read_csv(
        f"./results/result_merge/{project_name}_total.csv", low_memory=False
    )
    return pr_input


def get_pr_number(j):
    id = j["pr_url"].split("/")
    if "pulls" in id:
        id = id[id.index("pulls") + 1]
    elif "issues" in id:
        id = id[id.index("issues") + 1]
    else:
        _ = 5 / 0
    return id


def unify_pr_url(cur):
    if "issues" in cur["pr_url"]:
        cur["pr_url"].replace("issues", "pulls")
    if cur["pr_url"].split("/")[-1] == "commits":
        cur["pr_url"] = "/".join(cur["pr_url"].split("/")[:-1])
    return cur


def merge_features_in_projects(repo_list):
    for proj in repo_list:
        print(f"Working ON {proj['name']}")
        project_name = proj["name"].replace("/", "_")

        path = f"./results/result/{project_name}"
        proj_data = []

        for root, dirs, files in os.walk(path):
            # read all csv file under project file
            for f in files:
                file_name = os.path.join(root, f)
                # old no use merge info
                # if 'pr_merge_info.csv' in file_name:
                #     continue

                if "pr_comment_count_2_4.csv" in file_name:
                    continue

                if "pr_tags_2_5.csv" in file_name:
                    continue

                cur_data = read_csv_data_as_dict(file_name)
                mp = {}
                for j in cur_data:
                    id = get_pr_number(j)
                    mp[id] = j

                if len(proj_data) == 0:
                    proj_data = cur_data.copy()

                for i in range(len(proj_data)):
                    id1 = get_pr_number(proj_data[i])
                    if id1 in mp:
                        cur = mp[id1].copy()
                        cur = unify_pr_url(cur)
                        proj_data[i].update(cur)

        output_path = "./results/result_merge"
        mkdir(output_path)
        output_file = os.path.join(output_path, project_name + "_total.csv")
        write_csv_data_as_list_of_dict(output_file, proj_data)


def merge_all_projects(repo_list):
    one_set = None
    for i, proj in enumerate(repo_list):
        project_name = proj["name"].replace("/", "_")

        print("merging: ", project_name)
        project_data = read_data(project_name)

        if one_set is None:  # 475420 => 470821
            one_set = project_data.copy()
        else:
            one_set = pd.concat([one_set, project_data])[one_set.columns]
    one_set = one_set.dropna(axis=0)
    one_set.to_csv(
        "./results/all_projects_features_Z2.csv", index=False, sep=",", encoding="utf-8"
    )


def main():
    repo_list = read_json_data(config["repo_list_file"])
    merge_features_in_projects(repo_list)
    merge_all_projects(repo_list)


if __name__ == "__main__":
    main()
