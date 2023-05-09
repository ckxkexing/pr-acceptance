######################################
# 计时器
######################################

import time
class Timer():
    def __init__(self):
        self.time = time.time()

    def start(self):
        self.time = time.time()

    def stop(self):
        return time.time() - self.time

######################################
# 处理csv、json文件
######################################

import csv
import json
import os.path
def read_csv_data(file_name):
    with open(file_name, 'r') as f:
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

def read_csv_data_as_dict(file_name):
    '''
        以字典形式读取dict文件
    '''
    res = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            res.append(row)
    return res

def check_file_name(file_name):
    append_suffix = 0
    while(os.path.isfile(file_name)):
        arr = file_name.split(".")
        suffix = arr[-1]
        name = '.'.join(arr[0:-1])
        file_name = name + f'({append_suffix}).{suffix}'
        append_suffix += 1
    return file_name

def write_csv_data(csv_file_name, header, data):
    # csv_file_name = check_file_name(csv_file_name)
    with open(csv_file_name, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)

def read_json_data(json_file_name):
    with open(json_file_name, 'r') as f:
        data = json.load(f)
    return data

def write_json_data(data, json_file_name):
    # json_file_name = check_file_name(json_file_name)
    with open(json_file_name, 'w') as f:
        json.dump(data, f, indent=2)

def csv2json(csv_name, json_name=None):
    header, data = read_csv_data(csv_name)
    res = []
    for da in data:
        cur = {}
        for i , val in enumerate(da):
            cur[header[i]] = val
        res.append(cur)
    if json_name :
        with open(json_name, 'w') as f:
            json.dump(res, f, indent=2)
    else :
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
            else :
                cur.append(None)
        res.append(cur)
    if csv_name :
        write_csv_data(csv_name, header, res)
    else :
        return header, res
