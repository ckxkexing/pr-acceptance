from use_ghtorrent.utils import read_json_data

from use_ghtorrent.user import *
from use_other.user_type import *
import os

def main():
    data = read_json_data('star_proj_get_2021_9_7_pre50.json')
    flag = 1
    for proj in data:
        # if proj['name'] in ['kubernetes/kubernetes', 'vercel/next.js']:  # 
        #     continue
        if proj['name'] == 'vercel/next.js':
            continue

        print(f"正在处理{proj['name']}")
        project_name = proj['name'].replace('/', '_')

        json_file = f'/data1/chenkexing/star_50/human_readable_pulls/human_pull_{project_name}.json'
        path = f'results/result/{project_name}'
        if not os.path.exists(path):
            os.makedirs(path)

        # 1.2
        # xxx_before_pr_user_projects_1_2.csv
        # cal_before_pr_user_project(json_file, project_name, path)

        # 1.4
        # pytorch_pytorch_before_pr_user_commits_proj_1_4.csv
        # moby/moby pymysql : 6900 s // mysqlclient : 137 s
        # cal_before_pr_user_commits_project(json_file, project_name, path)

        # 1.5
        # cal_before_pr_user_commits(json_file, project_name, path)

        # 1.6
        # cal_before_pr_user_prs(json_file, project_name, path)

        # 1.7
        # cal_before_pr_user_issues(json_file, project_name, path)

        # 1.8
        ## cal_before_pr_user_followers(json_file, project_name, path)
        # break

        # detect user type
        # bot_detect_by_login(json_file, project_name, path)


if __name__ == '__main__' :
    main()