'''
    training E-PRedict(Bert + XGBoost)
    train set and test set in different path
'''

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast

from models import (CodeBERTClassifer, VDCNNClassifer) 

from transformers import RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import set_seed
from transformers.optimization import get_linear_schedule_with_warmup

from utils import set_random_seed, save_checkpoint, load_checkpoint
from utils import AverageMeter, Res, mkdir
from utils import label_smoothing, str2bool, adjust_learning_rate

from xgboost import XGBClassifier

import logging

from arguments import args
from train_function import train_deep, val_deep, train_xgboost, val_xgboost
from datasets import load_datasets, load_datasets_by_project

def worker_main(args, logger):

    if args.seed:
        set_random_seed(args.seed, False)
        set_seed(args.seed)

    if not args.test_on_project or args.test_on_project == "None":
        train_dataset, valid_dataset, test_dataset = load_datasets(args, logger)
    else:
        train_dataset, valid_dataset, test_dataset = load_datasets_by_project(args, logger)

    trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=None,
        drop_last=True)

    valloader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.batch_size_per_gpu,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            sampler=None,
            drop_last=False)

    testloader = DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size_per_gpu,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            sampler=None,
            drop_last=False)

    logger.info("model initing")
    if 'BERT' in args.model:
        model = globals()[args.model](args, text_tokenizer_len=train_dataset.text_tokenizer_len, diff_tokenizer_len=train_dataset.diff_tokenizer_len, n_classes=2)
        model.cuda()
        optimizer = AdamW(model.parameters(),
                            lr = args.learning_rate, eps = 1e-8)
        t_total = len(trainloader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = t_total * 0.1,
                                        num_training_steps = t_total)
        scaler = GradScaler()
    elif args.model == 'VDCNNClassifer':
        model = VDCNNClassifer(args, n_classes=2)
        model.cuda()
        optimizer = optim.SGD(model.parameters(),
            lr = args.learning_rate, weight_decay=3e-4, momentum=0.9)
    elif args.model == 'baseline':
        model = None
        optimizer = None
        scheduler = None
        scaler = None

    if args.phase == 2:
        logger.info("%s", "Train Without Loading From Phase 1 Parameters")

    best_auc = 0
    for epoch in range(args.epochs):
        if args.model != "baseline":
            train_deep(model, optimizer, scheduler, scaler, trainloader, epoch, logger, args)

        # if (epoch + 1) % 3 == 0:
        #     logger.info("only use textual input's encode to val!")
        #     val_deep(model, valloader, epoch, logger, args)
        clf = train_xgboost(model, trainloader, epoch, logger, args)
        
        acc_1, prec_1, kappa_1, recall_1, f1_1, auc_1, acc_0, prec_0, kappa_0, recall_0, f1_0, auc_0 = \
            val_xgboost(model, clf, valloader, epoch, logger, args)
        
        if args.save_model and best_auc < auc_1:
            best_auc = auc_1
            mkdir(args.save_dir)
            save_checkpoint(args.save_dir, model, optimizer, clf)

    # Test phase.
    if args.test and args.save_model:
        loaded = load_checkpoint(args.save_dir, model, xgb_model=True)

        if 'model' in loaded:
            model = loaded['model']
        else:
            model = None

        if 'xgb_model' in loaded:
            clf = loaded['xgb_model']
        else:
            clf = None

        val_xgboost(model, clf, testloader, "Test Phase", logger, args)

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(args.log_name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("-" * 10 + "start-NEW-training" + "-" * 10)
    worker_main(args, logger)
