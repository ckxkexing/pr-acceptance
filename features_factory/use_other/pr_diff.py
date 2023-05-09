import os
from tqdm import tqdm
from .utils import read_json_data, write_csv_data, mkdir

def collect_pr_diff(file_dir, project_name, csv_file_path):
    pr_data = read_json_data(file_dir)
    header = ['pr_url', 'diff']
    res = []
    for data in tqdm(pr_data):

        cur = []
        cur.append(data['url'])
        
        # 
        # 
        # 

        res.append(cur)


    output_file = os.path.join(csv_file_path, f"{project_name}_pr_diff.csv")
    write_csv_data(output_file, header, res)
    