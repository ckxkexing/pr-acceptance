'''
    using k-fold validation
    to test PRdictor(bert + xgboost)
'''
import logging

import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import set_seed
from transformers.optimization import get_linear_schedule_with_warmup

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

from xgboost import XGBClassifier

from utils import Res
from utils import set_random_seed, save_checkpoint, load_checkpoint, mkdir

from arguments import args
from models import CodeBERTClassifer,VDCNNClassifer
from train_function import train_deep, val_deep, train_xgboost, val_xgboost
from datasets import csvDataset, load_data, standardiseNumericalFeats

def worker_main(args, logger):

    if args.seed:
        set_random_seed(args.seed, False)
        set_seed(args.seed)

    df_dataset = load_data(args.train_dir, project_name = args.train_test_on_project)
    print(len(df_dataset))
    res1 = Res('Test set Merge base!')
    res0 = Res('Test set Reject base!')

    if args.time_series:
        kf = TimeSeriesSplit(n_splits=10)
    else:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for _fold, (train_index, val_index) in enumerate(kf.split(df_dataset)):
        print("#"*10, f"fold = {_fold}", "#"*10)
        logger.info("%s\n%s\n%s", "#"*10, f"fold = {_fold}", "#"*10)

        train_data = df_dataset.iloc[train_index].copy()
        val_data =  df_dataset.iloc[val_index].copy()
        val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=47)

        train_data, test_data, val_data = standardiseNumericalFeats(train_data, test_data, val_data)
        train_dataset = csvDataset(args, X=train_data, max_length=args.max_length)
        valid_dataset = csvDataset(args, X=val_data,  max_length=args.max_length)
        test_dataset  = csvDataset(args, X=test_data, max_length=args.max_length)        

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

        print("model initing")
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
            scheduler = None
            scaler = None
        elif args.model == 'baseline':
            model = None
            optimizer = None
            scheduler = None
            scaler = None            
        best_auc = 0
        for epoch in range(args.epochs):
            if args.model != "baseline":
                train_deep(model, optimizer, scheduler, scaler, trainloader, epoch, logger, args)

            clf = train_xgboost(model, trainloader, epoch, logger, args)

            acc_1, prec_1, kappa_1, recall_1, f1_1, auc_1, acc_0, prec_0, kappa_0, recall_0, f1_0, auc_0 = \
                val_xgboost(model, clf, valloader, epoch, logger, args)
    
            if args.save_model and best_auc < auc_1:
                best_auc = auc_1
                mkdir(args.save_dir)
                save_checkpoint(args.save_dir, model, optimizer, clf)

        # Test phase.
        cur = [0] * 12
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

            acc_1, prec_1, kappa_1, recall_1, f1_1, auc_1, acc_0, prec_0, kappa_0, recall_0, f1_0, auc_0 = \
                val_xgboost(model, clf, testloader, "Test Phase", logger, args)
            cur = [acc_1, prec_1, kappa_1, recall_1, f1_1, auc_1, acc_0, prec_0, kappa_0, recall_0, f1_0, auc_0]
            res1.update(*cur[0:6])
            res0.update(*cur[6:])
            res = f'pre fold-{_fold} ave result:' + '\n' + res1.__str__() + '\n' + res0.__str__()
            print(res)
            logger.info(res)
        # clear
        if model:
            model.cpu()


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