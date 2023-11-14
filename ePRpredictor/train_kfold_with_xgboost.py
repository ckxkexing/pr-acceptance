"""
    using k-fold validation
    to test PRdictor(bert + xgboost)
"""
import logging

from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from transformers import set_seed

from arguments import args
from datasets import csvDataset, load_data, standardiseNumericalFeats
from train_function import (
    init_loader,
    init_model,
    test_phase,
    train_deep,
    train_xgboost,
    val_xgboost,
)
from utils.calculate import Res, set_random_seed
from utils.file import mkdir
from utils.model_save import save_checkpoint


def worker_main(args, logger):
    if args.seed:
        set_random_seed(args.seed, False)
        set_seed(args.seed)

    df_dataset = load_data(args.train_dir, project_name=args.train_test_on_project)
    print(len(df_dataset))
    res1 = Res("Test set Merge base!")
    res0 = Res("Test set Reject base!")

    if args.time_series:
        kf = TimeSeriesSplit(n_splits=10)
    else:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for _fold, (train_index, val_index) in enumerate(kf.split(df_dataset)):
        print("#" * 10, f"fold = {_fold}", "#" * 10)
        logger.info("%s\n%s\n%s", "#" * 10, f"fold = {_fold}", "#" * 10)

        train_data = df_dataset.iloc[train_index].copy()
        val_data = df_dataset.iloc[val_index].copy()
        val_data, test_data = train_test_split(val_data, test_size=0.5, random_state=47)

        train_data, test_data, val_data = standardiseNumericalFeats(
            train_data, test_data, val_data
        )
        train_dataset = csvDataset(args, X=train_data, max_length=args.max_length)
        valid_dataset = csvDataset(args, X=val_data, max_length=args.max_length)
        test_dataset = csvDataset(args, X=test_data, max_length=args.max_length)

        train_loader, val_loader, test_loader = init_loader(
            args, train_dataset, valid_dataset, test_dataset
        )

        model, optimizer, scheduler, scaler = init_model(
            args, train_dataset, train_loader
        )

        best_auc = 0
        for epoch in range(args.epochs):
            if args.model != "baseline":
                train_deep(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    train_loader,
                    epoch,
                    logger,
                    args,
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
            cur = test_phase(args, model, test_loader, logger)

            res1.update(*cur[0:6])
            res0.update(*cur[6:])
            res = (
                f"pre fold-{_fold} ave result:"
                + "\n"
                + res1.__str__()
                + "\n"
                + res0.__str__()
            )
            logger.info(res)

        if model:
            model.cpu()


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
