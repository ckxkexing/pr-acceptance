"""
    training E-PRedict(Bert + XGBoost)
    train set and test set in different path
"""

import logging

from transformers import set_seed

from arguments import args
from datasets import load_datasets, load_datasets_by_project
from models import CodeBERTClassifer, VDCNNClassifer  # noqa
from train_function import (
    init_loader,
    init_model,
    test_phase,
    train_deep,
    train_xgboost,
    val_xgboost,
)
from utils.calculate import set_random_seed
from utils.file import mkdir
from utils.model_save import save_checkpoint


def worker_main(args, logger):
    if args.seed:
        set_random_seed(args.seed, False)
        set_seed(args.seed)

    if not args.test_on_project or args.test_on_project == "None":
        train_dataset, valid_dataset, test_dataset = load_datasets(args, logger)
    else:
        train_dataset, valid_dataset, test_dataset = load_datasets_by_project(
            args, logger
        )

    train_loader, val_loader, test_loader = init_loader(
        args, train_dataset, valid_dataset, test_dataset
    )

    model, optimizer, scheduler, scaler = init_model(args, train_dataset, train_loader)

    if args.phase == 2:
        logger.info("%s", "Train Without Loading From Phase 1 Parameters")

    best_auc = 0
    for epoch in range(args.epochs):
        if args.model != "baseline":
            train_deep(
                model, optimizer, scheduler, scaler, train_loader, epoch, logger, args
            )

        clf = train_xgboost(model, train_loader, epoch, logger, args)

        (
            acc_1,
            prec_1,
            kappa_1,
            recall_1,
            f1_1,
            auc_1,
            acc_0,
            prec_0,
            kappa_0,
            recall_0,
            f1_0,
            auc_0,
        ) = val_xgboost(model, clf, val_loader, epoch, logger, args)

        if args.save_model and best_auc < auc_1:
            best_auc = auc_1
            mkdir(args.save_dir)
            save_checkpoint(args.save_dir, model, optimizer, clf)

    if args.test and args.save_model:
        test_phase(args, model, test_loader, logger)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("-" * 10 + "start-NEW-training" + "-" * 10)
    worker_main(args, logger)
