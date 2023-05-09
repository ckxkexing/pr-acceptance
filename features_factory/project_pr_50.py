'''
    处理50个项目的所有project的所有pr信息
'''
import os
from use_ghtorrent.project_pr import *
from use_other.project_pr import *
from utils import read_json_data


def main():
    data = read_json_data('star_proj_get_2021_9_7_pre50.json')
    total = len(data)
    cnt = 0
    for proj in data:
        cnt += 1
        # tmp
        if proj['name'] == 'vercel/next.js':
            continue
        # tmp

        #### 使用 ####
        print(f"[{cnt} / {total} ]正在处理{proj['name']}")
        repo_name = proj['name']
        project_name = proj['name'].replace('/', '_')

        path = f'results/result/{project_name}'
        if not os.path.exists(path):
            os.makedirs(path)

        json_file = f'/data1/chenkexing/star_50/human_readable_pulls/human_pull_{project_name}.json'
        pulls_json_file = f'/data1/chenkexing/star_50/human_readable_pulls/human_pull_{project_name}.json'
        issue_json_file = f'/data1/chenkexing/star_50/issues/{project_name}_repos_issues.jsonlines'
        clone_dir = '/data1/chenkexing/star_50/clone_star_50_projects_only_commits'
        repo_clone_dir = os.path.join(clone_dir, repo_name) + '/.git'
        merge_info_dir = os.path.join(path, f"{project_name}_merge_info.csv")
        # 4.3
        # print("[ - ] 4.3 PR提交前commits个数")
        # tongji_project_before_pr_commits(pulls_json_file, repo_name, project_name, path)
        # tongji_project_before_pr_commits_in_cloned(pulls_json_file, repo_clone_dir, repo_name, project_name, path)

        # 4.4
        # print("[ - ] 4.4 pr提交前一个月commits个数")
        # tongji_project_before_pr_commits_in_months(pulls_json_file, repo_name, project_name, path)
        # tongji_project_before_pr_commits_in_months_in_cloned(pulls_json_file, repo_clone_dir, repo_name, project_name, path)
        
        # 4.5
        # print("[ - ] 4.5 pr提交前的comment数目")
        # tongji_project_before_pr_comments(pulls_json_file, repo_name, project_name, path)
        # tongji_project_before_pr_comments_use_mongo(pulls_json_file, issue_json_file, repo_name, project_name, path)

        # 4.6
        # print("[ - ] 4.6 PR提交前一个月的comment数目")
        # tongji_project_before_pr_comments_in_months(pulls_json_file, repo_name, project_name, path)
        # tongji_project_before_pr_comments_in_months_use_mongo(pulls_json_file, issue_json_file, repo_name, project_name, path)

        # 4.7、4.8
        # print("[ - ] 4.7、4.8 PR提交前的pr数目、提交前一个月内的pr数目")
        # tongji_project_before_pr_prs(pulls_json_file, repo_name, project_name, path)

        # 4.9、4.10
        # print("[ - ] 4.9、4.10 PR提交前的issue_comment数目、提交前一个月的issue_comment数目")
        # tongji_project_before_pr_issues(pulls_json_file, repo_name, project_name, path)
        # tongji_project_before_pr_issues_use_scrapy(pulls_json_file, issue_json_file ,repo_name, project_name, path)

        # 4.11
        tongji_project_before_pr_merge_ratio(pulls_json_file, merge_info_dir, repo_name, project_name, path)
if __name__ == '__main__' :
    main()