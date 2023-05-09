'''

    read feature.csv for baseline text

'''
import os
import logging
import argparse

from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_curve, auc

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd
from utils import AverageMeter
from collections import defaultdict

import matplotlib.pyplot as plt

from tqdm import tqdm
from math import sqrt
from utils import Res
from utils import str2bool
from utils import mkdir
from train_function import gen_clf

def get_arguments():
    """
        Parse all the arguments
        Returns: args
        A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="baseline")
    parser.add_argument("--classifier", type=str, dest='classifier', default='bert')
    parser.add_argument("--input_dir", type=str, dest='input_dir', default=None)
    parser.add_argument("--nominize_in_project", type=str2bool, default='False', help="check for nominize") 
    parser.add_argument("--project_name", type=str, dest='project_name', default=None)
    parser.add_argument("--time_series", type=str2bool, dest='time_series', default='False')
    parser.add_argument("--log_importance_photo", type=str, dest='log_importance_photo', default=None)
    parser.add_argument("--add_text_data", type=str2bool, dest='add_text_data', default='False') 
    parser.add_argument("--save_dir", default="", type=str, help="save dir") 
    parser.add_argument("--log_name", type=str, dest='log_name', default=None)
    args = parser.parse_args()
    mkdir(args.save_dir)
    args.log_name = os.path.join(args.save_dir, args.log_name)
    if args.log_importance_photo and args.log_importance_photo != "None":
        args.log_importance_photo = os.path.join(args.save_dir, args.log_importance_photo)
    return args

def standardiseNumericalFeats(X_train, X_test, X_val=None):
    """Standardise the numerical features
    
        Returns:
            Standardised X_train and X_test
    """

    numerical_cols = ['number_of_created_pr_in_this_proj_before_pr', 'commit_add_line_sum', 
        'commit_delete_line_sum', 'commit_total_line_sum', 'commit_file_change', 'commit_add_line_max', 'commit_delete_line_max', 
        'commit_total_line_max', 'commit_add_line_min', 'commit_delete_line_min', 'commit_total_line_min', 'before_pr_project_commits_in_month', 
        'issue_created_in_project_by_pr_author', 'issue_created_in_project_by_pr_author_in_month', 'issue_joined_in_project_by_pr_author', 
        'issue_joined_in_project_by_pr_author_in_month', 'before_pr_user_followers', 'before_pr_user_commits_proj', 'before_pr_project_issues', 
        'before_pr_project_issues_in_month', 'before_pr_project_issues_comment', 'before_pr_project_issues_comment_in_month', 'pr_commit_count', 'before_pr_project_prs', 
        'before_pr_project_prs_in_month', 'before_pr_project_commits', 'before_pr_user_commits', 'before_pr_user_issues', 
        'before_pr_merge_cnt', 'before_pr_closed_cnt', 'before_pr_project_comments_in_prs', 'before_pr_user_projects', 
        'number_of_merged_pr_in_this_proj_before_pr', 'number_of_closed_pr_in_this_proj_before_pr', 'before_pr_project_comments_in_prs_in_month', 
        'before_pr_user_pulls', 'pr_desc_len', 'contain_test_file', 'contain_doc_file']

    for col in numerical_cols:
        scaler = StandardScaler()
        if col not in list(X_train):
            continue
        X_train[col] = scaler.fit_transform(X_train[[col]])
        X_test[col] = scaler.transform(X_test[[col]])
        if X_val:
            X_val[col] = scaler.transform(X_val[[col]])
    return X_train, X_test, X_val


def data_load(train_file, add_text_data = True, project_name=None):
    def dropout(pr_data):
        for col in ['login', 'github_merge', 'comment_merge', 'commit_merge', 'closed_at']:
            if col not in list(pr_data):
                continue
            pr_data = pr_data.drop(columns=[col])

        # pr_data = pr_data.drop(columns=['pr_url'])

        for col in list(pr_data):
            if 'lifetime' in col:
                pr_data = pr_data.drop(columns=[col])

        if not add_text_data:
            pr_data = pr_data.drop(columns=['title', 'body'])
        pr_data = pr_data.drop(columns=['commit_message'])
        pr_data = pr_data.drop(columns=['code_patch_diff'])
        return pr_data

    pr_data = pd.read_csv(train_file, low_memory=False)
    if project_name:
        pr_data = pr_data[pr_data["pr_url"].str.contains(project_name)].copy()

    pr_data = dropout(pr_data)

    for col in ['code_patch_diff', 'commit_message',  'title', 'body']:
        if col in list(pr_data):
            pr_data[col] = pr_data[col].fillna('None')
    
    pr_data = pr_data.fillna(0)
    
    for col in list(pr_data):
        if (col not in ['pr_url', 'merge', 'code_patch_diff', 'commit_message', 'title', 'body']):
            pr_data[col] = pr_data[col].astype(np.float32)
    return pr_data


def data_processing(x_train, x_val, x_test, y_train, y_val, y_test, add_text_data = True):
    
    if add_text_data:
        ## add text data
        # stop_words = get_stop_words()
        # x_train = x_train.apply(lambda x: processing_sentence(x, stop_words))
        # x_test = x_test.apply(lambda x: processing_sentence(x, stop_words))

        # pr_data['commit_message'] = pr_data['commit_message'].apply(clean_text)

        tf = TfidfVectorizer(max_features = 5000, lowercase =True)

        sparse_a = tf.fit_transform((x_train['title'] + " <\n> " + x_train['body']).astype('U'))
        sparse_b = tf.transform((x_test['title'] + " <\n> " + x_test['body']).astype('U'))
        sparse_c = tf.transform((x_val['title'] + " <\n> " + x_val['body']).astype('U'))
        svd = TruncatedSVD(n_components=10, random_state=42)

        dense_a = svd.fit_transform(sparse_a)
        dense_b = svd.transform(sparse_b) 
        dense_c = svd.transform(sparse_c)
        dfa = pd.DataFrame(dense_a, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'])
        dfb = pd.DataFrame(dense_b, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'])
        dfc = pd.DataFrame(dense_c, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'])

        def concat(x, df):
            x.drop('title', axis=1, inplace=True)
            x.drop('body', axis=1, inplace=True)
            x.reset_index(drop=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            x = pd.concat([x, df], axis=1)
            return x
        x_train = concat(x_train, dfa)
        x_test  = concat(x_test, dfb)
        x_val   = concat(x_val, dfc)

    # x_train, x_test, x_val = standardiseNumericalFeats(x_train, x_test, x_val)
    columns = list(x_train)
    print("#" * 20)
    print("Feature counts = ", len(columns))
    print("#" * 20)
    # columns.extend(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K'])
    # return np.array(x_train), np.array(x_test), y_train, y_test, columns
    return x_train.to_numpy(), x_val.to_numpy(), x_test.to_numpy(), y_train, y_val, y_test, columns


def eval_pred(preds, labels):
    ### precision
    pre = precision_score(labels, preds)
    ### recall
    recall = recall_score(labels, preds)
    ### accuracy
    acc = accuracy_score(labels, preds)
    ### f1
    f1 = f1_score(labels, preds)
    ### kappa 
    kappa = cohen_kappa_score(labels, preds)
    return acc, pre, recall, f1, kappa


def Quantify_in_the_project(pr_data, feature_cnt=46):
    proj_feature_mean = defaultdict(lambda:[0] * feature_cnt)
    proj_feature_sd   = defaultdict(lambda:[0] * feature_cnt)
    project_cnt = defaultdict(lambda: 0)
    # For feature，Calculate average value by the projects
    for index, data in tqdm(pr_data.iterrows()):
        url = data[0]
        project_name = "_".join(url.split("/")[-4:-2])
        features = data.values[1:-1]
        features = features.tolist()
        proj_feature_mean[project_name] = \
            [i + j for i, j in zip(proj_feature_mean[project_name], features)]
        project_cnt[project_name] += 1
    for proj in proj_feature_mean:
        cnt = project_cnt[proj]
        proj_feature_mean[proj] = [value / cnt for value in proj_feature_mean[proj]]
    
    # Calculate standard deviation.
    for index, data in tqdm(pr_data.iterrows()):
        url = data[0]
        project_name = "_".join(url.split("/")[-4:-2])
        features = data.values[1:-1]
        features = features.tolist()
        proj_feature_sd[project_name] = \
            [i + (k - j)*(k - j)  for i, j, k in zip(proj_feature_sd[project_name], proj_feature_mean[project_name], features)]
    for proj in proj_feature_sd:
        cnt = project_cnt[proj]
        proj_feature_sd[proj] = [sqrt(value / cnt + 1e-12) for value in proj_feature_sd[proj]]
    
    new_col = [[] for i in range(feature_cnt)]
    for index, data in tqdm(pr_data.iterrows()):
        url = data[0]
        project_name = "_".join(url.split("/")[-4:-2])
        features = data.values[1:-1]
        features = features.tolist()
        features = [(i-j) / k for i, j, k in zip(features, proj_feature_mean[project_name], proj_feature_sd[project_name])]
        
        for id in range(0, feature_cnt):
            new_col[id].append(features[id])
    for id in range(0, feature_cnt):
        pr_data.iloc[:, id+1] = new_col[id]
    return pr_data

def output_importance_plt(mp_list, dir):
    x_ticks = [i+1 for i in range(10)]
    x = np.arange(len(x_ticks))

    plt.figure(figsize=(30, 18))

    for feature in mp_list:

        if np.max(mp_list[feature]) < 0.02:
            continue

        plt.plot(x, mp_list[feature], label=feature, linewidth=3.0)
        for a, b in zip(x, mp_list[feature]):
            plt.text(a, b, '%.3f'%b, ha='center', va= 'bottom', fontsize=18)
    
    plt.xticks([r for r in x], x_ticks, fontsize=18, rotation=20)
    plt.yticks(fontsize=18)

    plt.xlabel(u'x_label', fontsize=18)
    plt.ylabel(u'y_label', fontsize=18)

    plt.title(u'Title', fontsize=18)

    plt.legend()

    plt.savefig(dir, bbox_inches='tight')

def solve(args, logger):
    data_path = args.input_dir
    pr_data = data_load(data_path, args.add_text_data, args.project_name)
    # Move merge column to last.
    cols = list(pr_data.columns.values)
    cols.remove('merge')
    cols.append('merge')
    pr_data = pr_data[cols]

    # print(pr_data.loc[0])

    if args.nominize_in_project:
        pr_data = Quantify_in_the_project(pr_data)
    
    # print(pr_data.loc[0])

    res1 = Res('Merge base!')
    res0 = Res('Reject base!')
    ave_auc1 = AverageMeter('ave_auc1', ':.3f', 0)
    ave_auc0 = AverageMeter('ave_auc0', ':.3f', 0)
    
    if args.time_series:
        kf = TimeSeriesSplit(n_splits=10)
    else:
        kf = KFold(n_splits=10, shuffle=True, random_state=42)

    feature_importance_list = {}
    for k, (train_idx, val_idx) in enumerate(kf.split(pr_data)):

        train_set = pr_data.iloc[train_idx].copy()
        val_set =  pr_data.iloc[val_idx].copy()
        val_set, test_set = train_test_split(val_set, test_size=0.5, random_state=47)

        x_train, y_train = train_set.drop(columns=['merge', 'pr_url']), list(zip(train_set['merge'], train_set['pr_url']))
        x_val  , y_val   = val_set.drop(columns=['merge', 'pr_url']), list(zip(val_set['merge'], val_set['pr_url']))
        x_test, y_test   = test_set.drop(columns=['merge', 'pr_url']), list(zip(test_set['merge'], test_set['pr_url']))

        x_train, x_val, x_test, y_train, y_val, y_test, columns = data_processing(x_train, x_val, x_test, y_train, y_val, y_test, add_text_data = args.add_text_data)

        logger.info(f"{k} fold - Training!")

        ####################
        # XGBoost
        # LogisticRegression
        # DecisionTreeClassifier
        # RandomForest
        # DecisionTree
        ####################
        clf = gen_clf(args)
        clf.fit(x_train, [i[0] for i in y_train])

        logger.info("########## Cur Importance ##########")
        importance = sorted(zip(clf.feature_importances_, columns), reverse=True)
        for elm in importance:
            logger.info(elm)
            v, name = elm
            if name not in feature_importance_list:
                feature_importance_list[name] = []
            feature_importance_list[name].append(v)

        logger.info("########## Cur Importance ##########")

        logger.info(f"{k} fold - evaluing!")
        y_val_predict = clf.predict(x_val)
        y_predict = clf.predict(x_test)

        pred = np.array([[1] if x > 0 else [0] for x in y_predict])
        label = np.array([[1] if x[0] > 0 else [0] for x in y_test])

        logger.info(f"True label = {sum(label)}")
        logger.info(f"False label = {len(label) - sum(label)}" )
        
        acc, pre, recall, f1, kappa = eval_pred(pred, label)
        acc0, pre0, recall0, f10, kappa0 = eval_pred(pred==0, label==0) 
        logger.info(f'acc\t {acc}\t pre\t {pre}\t recall\t {recall}\t f1 {f1}\t kappa {kappa}')
        logger.info(f'acc\t {acc0}\t pre\t {pre0}\t recall\t {recall0}\t f1 {f10}\t kappa {kappa0}')
        
        fpr,tpr,thresholds = roc_curve(label, clf.predict_proba(x_test)[:, 1], pos_label=1)
        auc1 = auc(fpr,tpr)
        ave_auc1.update(auc1)
        logger.info(f'auc\t {auc1}')
        fpr,tpr,thresholds = roc_curve(label, clf.predict_proba(x_test)[:, 0], pos_label=0)
        auc0 = auc(fpr,tpr)
        ave_auc0.update(auc0)

        res1.update(acc, pre, kappa, recall, f1)
        res0.update(acc0, pre0, kappa0, recall0, f10 )

    logger.info(f"\n{res1.__str__()}\n{res0.__str__()}")
    logger.info(f"AVE-auc1 = { ave_auc1.avg:.3f}")
    logger.info(f"AVE-auc0 = { ave_auc0.avg:.3f}")

    if args.log_importance_photo and args.log_importance_photo != "None":
        output_importance_plt(feature_importance_list, args.log_importance_photo)
if __name__ == '__main__':
    args = get_arguments()

    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(args.log_name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("-" * 10 + "start-NEW-training" + "-" * 10)
    
    solve(args, logger)