import os
import random

import numpy as np

# set random seed
import torch
from torch.utils.data import DataLoader


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Res:
    """
    save result
    """

    def __init__(self, name):
        self.name = name
        self.ave_pre = AverageMeter("ave_pre", ":.3f", 0)
        self.ave_recall = AverageMeter("ave_recall", ":.3f", 0)
        self.ave_acc = AverageMeter("ave_acc", ":.3f", 0)
        self.ave_f1 = AverageMeter("ave_f1", ":.3f", 0)
        self.ave_kappa = AverageMeter("ave_kappa", ":.3f", 0)
        self.ave_auc = AverageMeter("ave_auc", ":.3f", 0)

    def update(self, acc, pre, kappa, recall, f1, auc=0):
        self.ave_acc.update(acc)
        self.ave_pre.update(pre)
        self.ave_recall.update(recall)
        self.ave_f1.update(f1)
        self.ave_kappa.update(kappa)
        self.ave_auc.update(auc)

    def __str__(self):
        return (
            f"{self.name}\t"
            f"acc:{self.ave_acc.avg:.3f}\t"
            f"pre:{self.ave_pre.avg:.3f}\t"
            f"recall:{self.ave_recall.avg:.3f}\t"
            f"f1:{self.ave_f1.avg:.3f}\t"
            f"kappa:{self.ave_kappa.avg:.3f}\t"
            f"auc:{self.ave_auc.avg:.3f}"
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", start_count_index=0):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def label_smoothing(inputs, epsilon=0.1):
    return ((1 - epsilon) * inputs) + (epsilon / 2)


class DataPrefetcher(object):
    """prefetch data."""

    def __init__(self, loader: DataLoader):
        if not torch.cuda.is_available():
            raise RuntimeError("Prefetcher needs CUDA, but not available!")
        self.loader = loader
        # self.device = device

    def __len__(self):
        return len(self.loader)

    def mv_cuda(self, x):
        if torch.is_tensor(x):
            return x.to(device="cuda", non_blocking=True)
        if isinstance(x, dict):
            for keys in x:
                x[keys] = x[keys].to(device="cuda", non_blocking=True)
        return x

    def __iter__(self):
        stream = torch.cuda.Stream()
        is_first = True
        data = None
        for next_data in iter(self.loader):
            with torch.cuda.stream(stream):
                next_data = [self.mv_cuda(x) for x in next_data]
            if not is_first:
                yield data
            else:
                is_first = False

            torch.cuda.current_stream().wait_stream(stream)
            data = next_data
        yield data


# Adjust lr.
def lr_poly(base_lr, iter, max_iter, power, min_lr):
    coeff = (1 - float(iter) / max_iter) ** power
    return (base_lr - min_lr) * coeff + min_lr


def adjust_learning_rate(optimizer, lr, power, min_lr, i_iter, total_steps):
    for pg in optimizer.param_groups:
        pg["lr"] = lr_poly(lr, i_iter, total_steps, power, min_lr)
    return optimizer.param_groups[0]["lr"]
