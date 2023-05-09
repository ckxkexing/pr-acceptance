import os
import csv
import sys
import json
import pandas as pd
from utils import read_csv_data_as_dict, write_csv_data_as_list_of_dict, read_json_data, mkdir

csv.field_size_limit(sys.maxsize)

def read_data(project_name):
    pr_input = pd.read_csv(f'./results/result_merge/{project_name}_total.csv', low_memory=False)
    return pr_input

def main():
    data = read_json_data('star_proj_get_2021_9_7_pre50.json')
    # step1 : merge features in every projects 
    for proj in data:

        print(f"正在处理{proj['name']}")
        project_name = proj['name'].replace('/', '_')

        # # ## tmp ##
        # if project_name != 'vercel_next.js':
        #     continue
        # # ## tmp ##

        path = f'./results/result/{project_name}'
        proj_data = []

        for root, dirs, files in os.walk(path):
            # read all csv file under project file
            for f in files:
                file_name = os.path.join(root, f)
                # old no use merge info
                # if 'pr_merge_info.csv' in file_name:
                #     continue

                if 'pr_comment_count_2_4.csv' in file_name:
                    continue

                if 'pr_tags_2_5.csv' in file_name:
                    continue
                
                # tmp
                # if 'pr_commit_message.csv' in file_name:
                #     continue
                # if 'pr_description.csv' in file_name:
                #     continue
                # tmp
                cur_data = read_csv_data_as_dict(file_name)
                mp = {}
                for j in cur_data:
                    id2 = j['pr_url'].split('/')
                    if 'pulls' in id2:
                        id2 = id2[id2.index('pulls') + 1]
                    elif 'issues' in id2:
                        id2 = id2[id2.index('issues') + 1]
                    else:
                        c = c / 0
                    mp[id2] = j

                if len(proj_data) == 0:
                    proj_data = cur_data.copy()

                for i in range(len(proj_data)):
                    id1 = proj_data[i]['pr_url'].split('/')
                    if 'pulls' in id1:
                        id1 = id1[id1.index('pulls') + 1]
                    elif 'issues' in id1:
                        id1 = id1[id1.index('issues') + 1]
                    else:
                        c = c / 0
                    if id1 in mp:
                        cur = mp[id1].copy()
                        if 'issues' in cur['pr_url']:
                            cur['pr_url'].replace("issues", "pulls")
                        if cur['pr_url'].split('/')[-1] == 'commits':
                            cur['pr_url'] = '/'.join(cur['pr_url'].split('/')[:-1])
                        proj_data[i].update(cur)

        output_path = './results/result_merge'
        mkdir(output_path)
        output_file = os.path.join(output_path, project_name + '_total.csv')
        write_csv_data_as_list_of_dict(output_file, proj_data)

    # step2: merge all _total.csv
    one_set = None
    for i , proj in enumerate(data):
        project_name = proj['name'].replace("/", '_')

        print("merging: ", project_name)
        project_data = read_data(project_name)
        
        if one_set is None: # 475420 => 470821
            one_set = project_data.copy()
        else:
            one_set = pd.concat([one_set , project_data])[one_set.columns]
    one_set = one_set.dropna(axis=0)
    one_set.to_csv("./results/all_projects_features_Z2.csv", index=False, sep=',', encoding='utf-8')
    # one_set.to_csv("/data1/chenkexing/all_projects_features_Z2.csv",index=False,sep=',')
    

if __name__ == '__main__':
    main()