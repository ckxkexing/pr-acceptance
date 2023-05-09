
######################################
# deal csv、json file
######################################

import csv
import json
import os.path

def read_csv_data(file_name):
    csv.field_size_limit(500 * 1024 * 1024)
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
    res = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            res.append(row)
    return res

def write_csv_data_as_list_of_dict(file_name , dicts):
    # file_name = check_file_name(file_name)
    keys = dicts[0].keys()
    with open(file_name, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, extrasaction='ignore') 
        dict_writer.writeheader()
        dict_writer.writerows(dicts)

def check_file_name(file_name):
    append_suffix = 1
    while(os.path.isfile(file_name)):
        arr = file_name.split(".")
        suffix = arr[-1]
        name = '.'.join(arr[0:-1])
        file_name = name + f'_{append_suffix}.{suffix}'
        append_suffix += 1
    return file_name

def write_csv_data(csv_file_name, header, data):
    # csv_file_name = check_file_name(csv_file_name)
    with open(csv_file_name, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        f_csv.writerows(data)

def write_json_data(data, json_file_name):
    # json_file_name = check_file_name(json_file_name)
    with open(json_file_name, 'w') as f:
        json.dump(data, f, indent=2)

# set random seed
import torch
import random
import numpy as np
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    os.environ['PYTHONHASHSEED'] =str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Res():
    '''
        save result
    '''
    def __init__(self, name):
        self.name = name
        self.ave_pre = AverageMeter('ave_pre', ':.3f', 0)
        self.ave_recall = AverageMeter('ave_recall', ':.3f', 0)
        self.ave_acc = AverageMeter('ave_acc', ':.3f', 0)
        self.ave_f1 = AverageMeter('ave_f1', ':.3f', 0)
        self.ave_kappa = AverageMeter('ave_kappa', ':.3f', 0)
        self.ave_auc   = AverageMeter('ave_auc', ':.3f', 0)

    def update(self, acc, pre, kappa, recall, f1, auc=0):
        self.ave_acc.update(acc)
        self.ave_pre.update(pre)
        self.ave_recall.update(recall)
        self.ave_f1.update(f1)
        self.ave_kappa.update(kappa)
        self.ave_auc.update(auc)

    def __str__(self):
        return f'{self.name}\t:acc={self.ave_acc.avg:.3f}\tpre:{self.ave_pre.avg:.3f}\trecall:{self.ave_recall.avg:.3f}\tf1:{self.ave_f1.avg:.3f}\tkappa:{self.ave_kappa.avg:.3f}\tauc:{self.ave_auc.avg:.3f}'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', start_count_index=0):
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


def label_smoothing(inputs, epsilon=0.1):
    return ((1 - epsilon) * inputs) + (epsilon / 2)

######################################
# model save and load
######################################
import pickle
def save_checkpoint(dir, model, optimizer, xgb_model=None):
    state_dict = {
        'model': model.state_dict() if model else {},
        'optimizer': optimizer.state_dict() if optimizer else {}}
    torch.save(state_dict, f"{dir}/bert.pth.tar")
    if xgb_model != None:
        pickle.dump(xgb_model, open(f"{dir}/xgb_pickle.dat", "wb"))

def load_checkpoint(dir, model, optimizer=None, xgb_model=None):
    if not os.path.exists(dir):
        assert 1==2 , ('Sorry, don\'t have checkpoint.pth file, continue training!')
        return
    checkpoint = torch.load(f"{dir}/bert.pth.tar")
    res = {}
    if model:
        model.load_state_dict(checkpoint['model'])
        res['model'] = model
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
        res['optimizer'] = optimizer
    if xgb_model:
        xgb_model = pickle.load(open(f"{dir}/xgb_pickle.dat", "rb"))
        res['xgb_model'] = xgb_model
    return res

# Parse bool var.
import argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import os 
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

# Adjust lr.
def lr_poly(base_lr, iter, max_iter, power, min_lr):
    coeff = (1 - float(iter) / max_iter)**power
    return (base_lr - min_lr) * coeff + min_lr

def adjust_learning_rate(optimizer, lr, power, min_lr, i_iter, total_steps):
    for pg in optimizer.param_groups:
        pg['lr'] = lr_poly(lr, i_iter, total_steps, power, min_lr)
    return optimizer.param_groups[0]['lr']


import torch
from torch.utils.data import DataLoader
class DataPrefetcher(object):
    """prefetch data."""

    def __init__(self, loader: DataLoader):
        if not torch.cuda.is_available():
            raise RuntimeError('Prefetcher needs CUDA, but not available!')
        self.loader = loader
        # self.device = device

    def __len__(self):
        return len(self.loader)

    def mv_cuda(self, x):
        if torch.is_tensor(x):
            return x.to(device='cuda', non_blocking=True)
        if isinstance(x,dict):
            for keys in x:
                x[keys] = x[keys].to(device='cuda', non_blocking=True)
        return x

    def __iter__(self):
        stream = torch.cuda.Stream()
        is_first = True

        for next_data in iter(self.loader):
            with torch.cuda.stream(stream):
                next_data = [
                    self.mv_cuda(x) for x in next_data
                ]
            if not is_first:
                yield data
            else:
                is_first = False

            torch.cuda.current_stream().wait_stream(stream)
            data = next_data
        yield data