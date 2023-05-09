'''
    用于论文中展示的commit信息。
'''
import csv
import json
from utils import read_json_data
def write_csv_data(csv_file_name, header, data):
    with open(csv_file_name, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)

commit_dir = '/data1/chenkexing/star_50/pull_commits3'

data = read_json_data('star_proj_get_2021_9_7_pre50.json')
flag = 1
header = ['proj_name', 'commits']
res = []
for proj in data:
    cur = []
    print(f"正在处理{proj['name']}")
    project_name = proj['name'].replace('/', '_')
    json_file = f'{commit_dir}/{project_name}_pr_commits.json'
    json_data = json.load(open(json_file, 'r'))
    cur.append(project_name)
    cnt = 0
    for info in json_data:
        cnt += len(info['commits'])
    cur.append(cnt)
    res.append(cur)

write_csv_data('proj_commits.csv', header, res)