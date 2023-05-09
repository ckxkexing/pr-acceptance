'''
    处理pr的commit特征信息
'''

from use_ghtorrent.utils import read_json_data

from use_ghtorrent.commit_code import *
import os


def main():
    data = read_json_data('star_proj_get_2021_9_7_pre50.json')
    flag = 1
    for proj in data:
        # tmp ##
        # if '.js' not in proj['name']:
        #     continue
        # if proj['name'] != 'vercel/next.js':
        #     continue
        # tmp ##
        print(f"正在处理{proj['name']}")
        project_name = proj['name'].replace('/', '_')
        path = f'results/result/{project_name}'
        if not os.path.exists(path):
            os.makedirs(path)

        json_file = f'/data1/chenkexing/star_50/human_readable_pulls/human_pull_{project_name}.json'
        commits_json_file = f'/data1/chenkexing/star_50/pull_commits_pre1/{project_name}_pr_commits.json'

        # 3.3-3.5
        # 处理pr的 commit code 增改行数信息
        # 利用了mongo中的commit detail数据
        tongji_project_commit_total_change(commits_json_file, project_name, path)


if __name__ == '__main__' :
    main()