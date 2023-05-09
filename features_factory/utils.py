import csv
import json
import os
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

def read_json_data(json_file_name):
    with open(json_file_name, 'r') as f:
        data = json.load(f)
    return data

def read_csv_data_as_dict(file_name):
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

def write_csv_data_as_list_of_dict(file_name , dicts):
    keys = list(dicts[0].keys())
    for d in dicts:
        for k in d:
            if k not in keys:
                keys.append(k)
    with open(file_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, extrasaction='ignore')
        dict_writer.writeheader()
        dict_writer.writerows(dicts)

def write_csv_data(csv_file_name, header, data):
    with open(csv_file_name, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)

def tree2pdf(clf, feature_names):
    # 最好保存成pdf格式，保证图片的清晰度
    from six import StringIO  
    import pydot 
    dot_data = StringIO() 
    tree.export_graphviz(clf, out_file=dot_data,feature_names=feature_names) 
    graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    graph[0].write_pdf("pr_merge.pdf") 

def mkdir(path):
	folder = os.path.exists(path)
 
	if not folder:        
		os.makedirs(path)
        

# 计算某个计量变量的平均时间
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':.2f', start_count_index=0):
        self.name = name
        self.fmt = fmt
        self.reset()
        self.start_count_index = start_count_index

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.count == 0:
            self.N = n

        self.val = val
        self.count += n
        if self.count > (self.start_count_index * self.N):
            self.sum += val * n
            self.avg = self.sum / (self.count - self.start_count_index * self.N)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
