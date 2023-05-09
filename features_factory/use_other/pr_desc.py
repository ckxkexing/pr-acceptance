import os
from tqdm import tqdm
from .utils import read_json_data, write_csv_data, mkdir

def collect_pr_desc(file_dir, project_name, csv_file_path):
    pr_data = read_json_data(file_dir)
    header = ['pr_url', 'body', 'title']
    res = []
    for data in tqdm(pr_data):
        cur = []
        cur.append(data['url'])
        if data['body']:
            cur.append('<nl>'.join(data['body'].splitlines()))
        else :
            cur.append("none")

        if data['title']:
            cur.append('<nl>'.join(data['title'].splitlines()))
        else :
            cur.append("none")
        res.append(cur)

    output_file = os.path.join(csv_file_path, f"{project_name}_pr_description.csv")
    write_csv_data(output_file, header, res)
    