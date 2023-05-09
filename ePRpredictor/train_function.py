from tqdm import tqdm
from torch.cuda.amp import autocast as autocast

import numpy as np
import torch
from xgboost import XGBClassifier

from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, auc, roc_auc_score

from utils import write_csv_data

'''
    Bert training related function.
'''

def train_deep(model, optimizer, scheduler, scaler, trainloader, epoch, logger, args):
    # print(f"start {epoch} training!")
    logger.info(f"start {epoch} training!")
    model.train()
    losses = []
    precisions = []
    bar = tqdm(trainloader, total=len(trainloader), dynamic_ncols = True)
    for _, data in enumerate(bar, 0):
        model.train()
        pr_url, pr_diff, label, origin, lens = data
        if 'BERT' in args.model:
            input = pr_diff['input_ids'].cuda()
            mask  = pr_diff['attention_mask'].cuda()
            token_type_ids = pr_diff["token_type_ids"].cuda()
            features = pr_diff['features'].cuda()
            with autocast(enabled=args.fp16):
                res = model(input, mask, token_type_ids, features, label.cuda())
                loss = res['loss']
                pred = res['pred']
            pred = pred.cpu()
        elif args.model == 'VDCNNClassifer':
            input = pr_diff['input_ids'].cuda()
            features = pr_diff['features'].cuda()
            with autocast(enabled=args.fp16):
                res = model(input, features, label.cuda())
                loss = res['loss']
                pred = res['pred']
            pred = pred.cpu()

        pred_choice = (pred > 0.5).long().squeeze()
        correct = pred_choice.eq(label).cpu().sum()
        optimizer.zero_grad()

        # Adjust learning rate.
        if 'BERT' in args.model:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.train()
            scheduler.step()
        else:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        precisions.append(correct / len(label))
        bar.set_description("epoch {}, iter {}, loss {}, prec {}".format(
            epoch, _, round(np.mean(losses),3), round(np.mean(precisions),3)))
        if _ % 1000 == 0:
            logger.info(f"train epoch {epoch}, iter {_} ,loss = {loss}, prec = {correct / len(label)}")


def val_deep(model, valloader, epoch, logger, args):
    # print(f"start {epoch} directly eval!")
    logger.info(f"start {epoch} directly eval!")
    model.eval()
    total = 0
    cnt = 0
    probs1, probs0  = [], []
    preds, labels = [], []
    pos_cnt = 0
    val_res_text = ""
    val_features = torch.tensor([])
    logger.info("%s val set length = %s", "#" * 10, len(valloader))
    for data in tqdm(valloader, dynamic_ncols = True):
        pr_url, pr_diff, label, origin, lens = data
        pos_cnt += sum(label > 0)
        total += label.shape[0]

        with torch.no_grad():
            if 'BERT' in args.model:
                input = pr_diff['input_ids'].cuda()
                mask  = pr_diff['attention_mask'].cuda()
                token_type_ids = pr_diff["token_type_ids"].cuda()
                features = pr_diff['features'].cuda()
                with autocast(enabled=args.fp16):
                    res = model(input, mask, token_type_ids, features, label.cuda())
                    loss = res['loss']
                    pred = res['pred'].cpu()

            elif args.model == 'VDCNNClassifer':
                input = pr_diff['input_ids'].cuda()
                features = pr_diff['features'].cuda()
                with autocast(enabled=args.fp16):
                    res = model(input, features, label.cuda())
                    loss = res['loss']
                    pred = res['pred'].cpu()

        prob1 = pred
        prob0 = 1 - pred
        pred = (pred > 0.5).long().squeeze()
        cnt += sum(pred == label)
        probs1.extend(prob1)
        probs0.extend(prob0)
        preds.extend(pred)
        labels.extend(label)
    labels = np.array(labels)
    preds = np.array(preds)
    def cal(preds, labels):
        acc = sum(preds == labels) / len(labels)    
        prec = precision_score(labels, preds)
        kappa = cohen_kappa_score(labels.cpu(), preds.cpu())
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)    
        return acc, prec, kappa, recall, f1
    
    probs1 = torch.tensor(probs1)
    probs0 = torch.tensor(probs0)

    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    
    # info of Merge
    acc_1, prec_1, kappa_1, recall_1, f1_1 = cal(preds, labels)
    # print("val pre text result:")
    logger.info("val pre text result:")
    # print(f"epoch {epoch} Merge - acc = {acc_1:.3f} - prec = {prec_1:.3f} - kappa = {kappa_1:.3f} - recall = {recall_1:.3f} - f1 = {f1_1:.3f}")
    logger.info(f"epoch {epoch} Merge - acc = {acc_1:.3f} - prec = {prec_1:.3f} - kappa = {kappa_1:.3f} - recall = {recall_1:.3f} - f1 = {f1_1:.3f}")
    val_res_text += f"Merge - acc = {acc_1:.3f} - prec = {prec_1:.3f} - kappa = {kappa_1:.3f} - recall = {recall_1:.3f} - f1 = {f1_1:.3f}"
    
    # AUC of label 1 
    fpr,tpr,thresholds = roc_curve(labels.cpu(), probs1.cpu(), pos_label=1)
    auc_1 = auc(fpr,tpr)
    # fpr,tpr,thresholds = roc_curve(labels.cpu(), probs1.cpu(), pos_label=1)
    # auc_1 = auc(fpr,tpr)
    # print(f"epoch {epoch} Merge - AUC = {auc_1:.3f}")
    logger.info(f"epoch {epoch} Merge - AUC = {auc_1:.3f}")
    # print("The ROC AUC score: ", str(round(roc_auc_score(labels.cpu(), probs1.cpu()), 4)))
    
    # info of Reject
    acc_0, prec_0, kappa_0, recall_0, f1_0 = cal(1 - preds, 1 - labels)
    # print(f"epoch {epoch} Reject - acc = {acc_0:.3f} - prec = {prec_0:.3f} - kappa = {kappa_0:.3f} - recall = {recall_0:.3f} - f1 = {f1_0:.3f}")
    logger.info(f"epoch {epoch} Reject - acc = {acc_0:.3f} - prec = {prec_0:.3f} - kappa = {kappa_0:.3f} - recall = {recall_0:.3f} - f1 = {f1_0:.3f}")
    
    val_res_text += f"\nReject - acc = {acc_0:.3f} - prec = {prec_0:.3f} - kappa = {kappa_0:.3f} - recall = {recall_0:.3f} - f1 = {f1_0:.3f}"
    
    # AUC of label 0
    fpr, tpr, thresholds = roc_curve(labels.cpu(), probs0.cpu(), pos_label=0)
    auc_0 = auc(fpr,tpr)
    # print(f"epoch {epoch} Reject - AUC = {auc_0:.3f}")
    logger.info(f"epoch {epoch} Reject - AUC = {auc_0:.3f}")
    # print("The ROC AUC score: ", str(round(roc_auc_score(labels.cpu(), probs0.cpu()), 4)))
    logger.info(f"The ROC AUC score: {str(round(roc_auc_score(labels.cpu(), probs0.cpu()), 4))}")
    # count val dataset
    # print(f"epoch {epoch} Val Message - pos cnt = {pos_cnt} as {pos_cnt / total :.3f}")
    logger.info(f"epoch {epoch} Val Message - pos cnt = {pos_cnt} as {pos_cnt / total :.3f}")
    val_info = f"Val Message - pos cnt = {pos_cnt} as {pos_cnt / total :.3f}"
    return acc_1, prec_1, kappa_1, recall_1, f1_1, auc_1, acc_0, prec_0, kappa_0, recall_0, f1_0, auc_0


def gen_clf(args):
    if args.classifier in ['CodeBERTClassifer', 'XGBClassifier', 'XGBoost'] :
        return XGBClassifier(max_depth = 6, eval_metric='logloss', use_label_encoder=False, seed=42)
            # (max_depth=10, learning_rate=0.1, n_estimators=200, n_jobs=5,
            #             silent=False, objective='binary:logistic', scale_pos_weight=0.5)
    if args.classifier == 'LogisticRegression':
        return LogisticRegression(random_state=42, solver='sag')
    if args.classifier == 'DecisionTreeClassifier':
        return tree.DecisionTreeClassifier(criterion='entropy',max_depth=19)
    if args.classifier in ['RandomForestClassifier', 'RandomForest'] :
        return RandomForestClassifier(n_estimators=10)
    if args.classifier == 'DecisionTree':
        return tree.DecisionTreeClassifier(criterion='entropy',max_depth=19)
    return None


def train_xgboost(model, trainloader, epoch, logger, args):
    logger.info(f"start {epoch} xgboost training!")
    if args.model != "baseline":
        model.eval()
    
    val_features = torch.tensor([])
    labels = torch.tensor([])
    bar = tqdm(trainloader,total=len(trainloader), dynamic_ncols = True)
    for _, data in enumerate(bar, 0):
        pr_url, pr_diff, label , origin, lens = data
        label = label.long().cpu()
        with torch.no_grad():
            if 'BERT' in args.model:
                input = pr_diff['input_ids'].cuda()
                mask  = pr_diff['attention_mask'].cuda()
                token_type_ids = pr_diff["token_type_ids"].cuda()
                features = pr_diff['features'].cuda()
                with autocast(enabled=args.fp16):
                    text_features = model(input, mask, token_type_ids, features, true_label=False, gen_feature=True)
                    features = torch.cat([text_features.cpu(), features.cpu()], 1)
                    val_features = torch.cat([val_features, features], 0)
            elif args.model == 'VDCNNClassifer':
                input = pr_diff['input_ids'].cuda()
                features = pr_diff['features'].cuda()
                with autocast(enabled=args.fp16):
                    text_features = model(input, features, true_label=False, gen_feature=True)
                    features = torch.cat([text_features.cpu(), features.cpu()], 1)
                    val_features = torch.cat([val_features, features], 0)
            elif args.model == 'baseline':
                # input = pr_diff['input']
                features = pr_diff['features']
                val_features = torch.cat([val_features, features.cpu()], 0)                
        labels = torch.cat([labels, label], 0)
    labels = np.array(labels)
    val_features = np.array(val_features)
    if not val_features.any():
        print("########## Please reduce batch size! ##########")
    clf = gen_clf(args)
    clf.fit(val_features, labels)
    return clf


def val_xgboost(model, clf, valloader, epoch, logger, args):
    logger.info(f"start {epoch} eval!")
    if args.model != "baseline":
        model.eval() 
    total = 0
    cnt = 0
    probs1, probs0  = [], []
    preds, labels = [], []
    pos_cnt = 0
    val_res_text = ""
    val_features = torch.tensor([])
    pr_urls = []
    logger.info("%s dataset length = %s", "#" * 10, len(valloader))
    bar = tqdm(valloader, total=len(valloader), dynamic_ncols = True)
    for data in bar:
        pr_url, pr_diff, label, origin, lens = data
        label = label.long().cpu()
        pos_cnt += sum(label > 0)
        total += label.shape[0]
        with torch.no_grad():
            if 'BERT' in args.model:
                input = pr_diff['input_ids'].cuda()
                mask  = pr_diff['attention_mask'].cuda()
                token_type_ids = pr_diff["token_type_ids"].cuda()
                features = pr_diff['features'].cuda()
                with autocast(enabled=args.fp16):
                    text_features = model(input, mask, token_type_ids, features, true_label=False, gen_feature=True)
                    features = torch.cat([text_features.cpu(), features.cpu()], 1)
                    val_features = torch.cat([val_features, features], 0)
            elif args.model == 'VDCNNClassifer':
                input = pr_diff['input_ids'].cuda()
                features = pr_diff['features'].cuda()
                with autocast(enabled=args.fp16):
                    text_features = model(input, features, true_label=False, gen_feature=True)
                    features = torch.cat([text_features.cpu(), features.cpu()], 1)
                    val_features = torch.cat([val_features, features], 0)
            elif args.model == 'baseline':
                # input = pr_diff['input']
                features = pr_diff['features']
                val_features = torch.cat([val_features, features], 0)
        labels.extend(label)
        pr_urls.extend(list(pr_url))
    labels = np.array(labels)
    val_features = np.array(val_features)
    preds = clf.predict(val_features)

    def cal(preds, labels):
        acc = sum(preds == labels) / len(labels)    
        prec = precision_score(labels, preds)
        kappa = cohen_kappa_score(labels.cpu(), preds.cpu())
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)    
        return acc, prec, kappa, recall, f1
    
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    # Info of Merge.
    acc_1, prec_1, kappa_1, recall_1, f1_1 = cal(preds, labels)
    # print("val pre text result:")
    logger.info("RESULTS:")
    # print(f"epoch {epoch} Merge - acc = {acc_1:.3f} - prec = {prec_1:.3f} - kappa = {kappa_1:.3f} - recall = {recall_1:.3f} - f1 = {f1_1:.3f}")
    logger.info(f"epoch {epoch} Merge - acc = {acc_1:.3f} - prec = {prec_1:.3f} - kappa = {kappa_1:.3f} - recall = {recall_1:.3f} - f1 = {f1_1:.3f}")
    val_res_text += f"Merge - acc = {acc_1:.3f} - prec = {prec_1:.3f} - kappa = {kappa_1:.3f} - recall = {recall_1:.3f} - f1 = {f1_1:.3f}"
    
    # AUC of label 1 
    fpr,tpr,thresholds = roc_curve(labels, clf.predict_proba(val_features)[:, 1], pos_label=1)
    auc_1 = auc(fpr,tpr)
    
    # fpr,tpr,thresholds = roc_curve(labels.cpu(), probs1.cpu(), pos_label=1)
    # auc_1 = auc(fpr,tpr)
    # print(f"epoch {epoch} Merge - AUC = {auc_1:.3f}")
    logger.info(f"epoch {epoch} Merge - AUC = {auc_1:.3f}")
    
    # info of Reject
    acc_0, prec_0, kappa_0, recall_0, f1_0 = cal(1 - preds, 1 - labels)
    # print(f"epoch {epoch} Reject - acc = {acc_0:.3f} - prec = {prec_0:.3f} - kappa = {kappa_0:.3f} - recall = {recall_0:.3f} - f1 = {f1_0:.3f}")
    logger.info(f"epoch {epoch} Reject - acc = {acc_0:.3f} - prec = {prec_0:.3f} - kappa = {kappa_0:.3f} - recall = {recall_0:.3f} - f1 = {f1_0:.3f}")
    val_res_text += f"\nReject - acc = {acc_0:.3f} - prec = {prec_0:.3f} - kappa = {kappa_0:.3f} - recall = {recall_0:.3f} - f1 = {f1_0:.3f}"
    
    # AUC of label 0
    fpr,tpr,thresholds = roc_curve(labels, clf.predict_proba(val_features)[:, 0], pos_label=0)
    auc_0 = auc(fpr,tpr)
    # print(f"epoch {epoch} Merge - AUC = {auc_0:.3f}")
    logger.info(f"epoch {epoch} Reject - AUC = {auc_0:.3f}")
    
    # count val dataset
    # print(f"epoch {epoch} Val Message - pos cnt = {pos_cnt} as {pos_cnt / total :.3f}")
    logger.info(f"epoch {epoch} Val Message - pos cnt = {pos_cnt} as {pos_cnt / total :.3f}")
    val_info = f"Val Message - pos cnt = {pos_cnt} as {pos_cnt / total :.3f}"

    if args.gen_test_output and args.gen_test_output!="None":

        res_header = ['pr_url', 'pred', 'label']
        res = []
        labels = labels.tolist()
        preds = preds.tolist()
        for i in range(len(labels)):
            cur = [pr_urls[i], preds[i], labels[i]]
            res.append(cur)
        write_csv_data(args.gen_test_output, res_header, res)

    return acc_1, prec_1, kappa_1, recall_1, f1_1, auc_1, acc_0, prec_0, kappa_0, recall_0, f1_0, auc_0