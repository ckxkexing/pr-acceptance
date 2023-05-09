from use_ghtorrent.utils import read_json_data

from use_ghtorrent.pr_info import *
from use_other.pr_info import *
from use_other.pr_cmt_message import *
from use_other.pr_desc import *
from use_other.pr_diff import *
import os

def main():
    data = read_json_data('star_proj_get_2021_9_7_pre50.json')
    flag = 1
    for proj in data:

        ##### tmp ######
        if proj['name'] == 'vercel/next.js':
            continue
        ##### tmp ######

        print(f"正在处理{proj['name']}")
        repo_name = proj['name']
        project_name = proj['name'].replace('/', '_')
        
        json_file = f'/data1/chenkexing/star_50/human_readable_pulls/human_pull_{project_name}.json'
        path = f'./results/result/{project_name}'
        commit_json_file = f'/data1/chenkexing/star_50/pull_commits_pre1/{project_name}_pr_commits.json'
        comment_json_file = f'/data1/chenkexing/star_50/pull_comment/{project_name}_pr_comments.json'
        issue_json_file = f'/data1/chenkexing/star_50/issues/{project_name}_repos_issues.jsonlines'
        merge_info_dir = os.path.join(path, f"{project_name}_merge_info.csv")
        # clone_dir = '/data1/chenkexing/star_50/clone_star_50_projects'
        clone_dir = 'tmp/clone_star_50_projects'
        repo_clone_dir = os.path.join(clone_dir, project_name.replace('_', '/')) + '/.git'        
        if not os.path.exists(path):
            os.makedirs(path)
        # 2.1
        # call_previous_pr(json_file, project_name, path)
        
        # # 2.1.b
        # call_previous_issue(json_file, repo_name, project_name, path)
        # call_previous_issue_use_mongo(json_file, issue_json_file, repo_name, project_name, path)
        
        # # 2.1.c 用户
        # call_previous_merged_closed_pr(json_file, merge_info_dir, repo_name, project_name, path)

        # # 2.2
        # call_pr_has_good_description(json_file, project_name, path) 

        # # 2.3
        # get_pr_commit_count(commit_json_file, project_name, path)

        # # 2.4
        # get_pr_comment_count(comment_json_file, project_name, path)

        # # 2.5
        # # pulls的tag数量
        # call_pr_tags(json_file, project_name, path)

        # # pr commit message
        # collect_pr_commit_message(commit_json_file, project_name, path)

        # # pr commit contain test or docs file
        find_out_commit_type(commit_json_file, project_name, path)

        # # pr description
        # collect_pr_desc(json_file, project_name, path)

        # TODO: # pr diff
        # collect_pr_diff(json_file, project_name, path)

        # # weather pr was created before commit
        # get_pr_commit_which_first_created(commit_json_file, project_name, path)  

        # Calculate PR's eigenvalue.
        # cal_pr_eigenvector_centrality(json_file, commit_json_file, repo_clone_dir, project_name, path)


if __name__ == '__main__' :
    main()