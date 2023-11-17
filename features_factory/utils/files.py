######################################
# 处理csv、json文件
######################################

import csv
import json
import os.path

import pydotplus
from sklearn.tree import export_graphviz


def read_csv_data(file_name):
    with open(file_name, "r") as f:
        f_csv = csv.reader(f)
        flag = True
        data = []
        for row in f_csv:
            if flag:
                flag = False
                header = row
            else:
                data.append(row)
    return header, data


def read_csv_data_as_dict(file_name: str):
    """
    以字典形式读取dict文件
    """
    res = []
    with open(file_name, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            res.append(row)
    return res


def write_csv_data_as_list_of_dict(file_name, dicts):
    keys = list(dicts[0].keys())
    for d in dicts:
        for k in d:
            if k not in keys:
                keys.append(k)
    with open(file_name, "w", newline="") as output_file:
        dict_writer = csv.DictWriter(output_file, keys, extrasaction="ignore")
        dict_writer.writeheader()
        dict_writer.writerows(dicts)


def check_file_name(file_name):
    append_suffix = 0
    while os.path.isfile(file_name):
        arr = file_name.split(".")
        suffix = arr[-1]
        name = ".".join(arr[0:-1])
        file_name = name + f"({append_suffix}).{suffix}"
        append_suffix += 1
    return file_name


def write_csv_data(csv_file_name, header, data):
    # csv_file_name = check_file_name(csv_file_name)
    with open(csv_file_name, "w") as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)


def tree2pdf(clf, feature_names):
    # 最好保存成pdf格式，保证图片的清晰度

    # Export resulting tree to DOT source code string
    dot_data = export_graphviz(
        clf, feature_names=feature_names, out_file=None, filled=True, rounded=True
    )

    # Export to pdf
    pydot_graph = pydotplus.graph_from_dot_data(dot_data)
    pydot_graph.write_pdf("tree.pdf")


def read_json_data(json_file_name):
    with open(json_file_name, "r") as f:
        data = json.load(f)
    return data


def read_jsonline_data(file_name):
    with open(file_name, "r") as f:
        for line in f:
            lin = line.strip()
            if not lin:
                continue
            lin_dict = json.loads(lin)
            yield lin_dict


def write_json_data(data, json_file_name):
    # json_file_name = check_file_name(json_file_name)
    with open(json_file_name, "w") as f:
        json.dump(data, f, indent=2)


def csv2json(csv_name, json_name=None):
    header, data = read_csv_data(csv_name)
    res = []
    for da in data:
        cur = {}
        for i, val in enumerate(da):
            cur[header[i]] = val
        res.append(cur)
    if json_name:
        with open(json_name, "w") as f:
            json.dump(res, f, indent=2)
    else:
        return header, res


def json2csv(json_name, csv_name=None):
    data_ori = read_json_data(json_name)
    header = []
    for data in data_ori:
        for k in data.keys():
            if k not in header:
                header.append(k)
    res = []
    for data in data_ori:
        cur = []
        for k in header:
            if k in data:
                cur.append(data[k])
            else:
                cur.append(None)
        res.append(cur)
    if csv_name:
        write_csv_data(csv_name, header, res)
    else:
        return header, res


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
