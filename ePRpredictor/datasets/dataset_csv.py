'''
    load data from json file
'''
import re
from sklearn import datasets 
import torch
from torch.utils import data
from utils import read_csv_data
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AdamW

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import pandas as pd
import numpy as np
import random
import math
from tqdm import tqdm
import json

from collections import defaultdict
from math import sqrt
from sklearn.preprocessing import StandardScaler

import spacy
import string

def standardiseNumericalFeats(X_train, X_test, X_val):
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
        X_val[col]  = scaler.transform(X_val[[col]])
    return X_train, X_test, X_val


# tokenization
# Download https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.4.0/en_core_web_md-3.4.0.tar.gz.
# pip install en_core_web_md-3.4.0
tok = spacy.load('en_core_web_md')
def sen_tokenize(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

class csvDataset(torch.utils.data.Dataset):
    """
        Load data from csv files.
    """
    def __init__(self, args, X, max_length = 500, vocab=None, vocab2index=None):
        super(csvDataset, self).__init__()

        self.data = X
        self.args = args
        self.inputs = []
        self.labels = []

        self.max_length = max_length

        if 'BERT' in args.model:
            self.text_tokenizer = RobertaTokenizer.from_pretrained(
                    args.desc_backbone, do_lower_case=False)
            self.diff_tokenizer = RobertaTokenizer.from_pretrained(
                    args.diff_backbone, do_lower_case=False)

            # Add remove/add Character
            self.text_tokenizer.add_special_tokens({'additional_special_tokens':["</->", "</+>", "</n>", "pull_id", "commit_id"]})
            self.text_tokenizer_len = len(self.text_tokenizer)

            self.diff_tokenizer.add_special_tokens({'additional_special_tokens':["</->", "</+>", "</n>", "pull_id", "commit_id"]})
            self.diff_tokenizer_len = len(self.diff_tokenizer)

        elif args.model == 'VDCNNClassifer':
            self.vocabulary = list(""" abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
            self.vocab_len = len(self.vocabulary) + 1
            self.text_tokenizer = self.char_tokenizer
            self.diff_tokenizer = self.char_tokenizer
        elif args.model == 'LSTM':
            self.vocab = vocab
            self.vocab2index = vocab2index
            self.text_tokenizer = self.word_tokenizer
            self.diff_tokenizer = self.char_tokenizer

        self.dataset_setting = 'all projects pr data (desc + diff)'
        self.label0_cnt = sum(self.data['merge'] == 0)
        self.label1_cnt = sum(self.data['merge'] == 1)
        print("#" * 10, self.label0_cnt, "#" * 10, self.label1_cnt)

        self.columns = self.data.columns[~self.data.columns.isin(["pr_url", "title", "body", "merge", "commit_message", "code_patch_diff"])]

    def get_features_name(self):
        return list(self.columns)

    def char_tokenizer(self, input):
        input = [self.vocabulary.index(i) + 1 for i in list(input) if i in self.vocabulary]
        if len(input) < self.max_length:
            input += [0] * (self.max_length - len(input))
        if len(input) > self.max_length:
            input = input[:self.max_length]
        input = {'input_ids': torch.tensor(input).unsqueeze(0)}
        return input

    def word_tokenizer(self, input):
        tokenized = sen_tokenize(input)
        encoded = np.zeros(self.max_length, dtype=int)
        enc1 = np.array([self.vocab2index.get(word, self.vocab2index["UNK"]) for word in tokenized])
        length = min(self.max_length, len(enc1))
        encoded[:length] = enc1[:length]
        return {'input_ids': torch.tensor(encoded).unsqueeze(0)}

    def __getitem__(self, index):
        info = self.data.iloc[index]

        desc_title = info['title']
        desc_body  = info['body']
        pr_url     = info['pr_url']
        features = info[~self.data.columns.isin(["pr_url", "title", "body", "merge", "commit_message", "code_patch_diff"]) ]
        columns = self.data.columns[~self.data.columns.isin(["pr_url", "title", "body", "merge", "commit_message", "code_patch_diff"])]

        hide = self.args.hide
        # Hide some feature.
        if hide == 'hide_personal':
            # Hide Contributor Profile Dimension
            cols = ['before_pr_user_projects', 'before_pr_user_commits', 'before_pr_user_pulls', 'before_pr_user_issues', 'before_pr_user_followers', \
                    'before_pr_user_commits_proj', 'bot_user',  \
                    'issue_created_in_project_by_pr_author', 'issue_created_in_project_by_pr_author_in_month', 'issue_joined_in_project_by_pr_author', \
                    'issue_joined_in_project_by_pr_author_in_month',    
                    'number_of_created_pr_in_this_proj_before_pr', \
                    'ratio_of_merged_pr_in_this_proj_before_pr', \
                    'number_of_merged_pr_in_this_proj_before_pr', \
                    'number_of_closed_pr_in_this_proj_before_pr']
            for i, v in enumerate(columns.isin(cols)):
                if v : features[i] = 0
        elif hide == 'hide_pr_info':
            # Hide Specific Pull Request Dimension 
            cols = ['is_this_proj_first', 'pr_desc_len', 'check_pr_desc_mean', 'check_pr_desc_medium', 'pr_commit_count', \
                    'commit_add_line_sum', 'commit_add_line_max', 'commit_add_line_min', \
                    'commit_delete_line_sum', 'commit_delete_line_max', 'commit_delete_line_min', \
                    'commit_total_line_sum', 'commit_total_line_max', 'commit_total_line_min', \
                    'commit_file_change', 'whether_pr_created_before_commit', 'contain_test_file', 'contain_doc_file']
            for i, v in enumerate(columns.isin(cols)):
                if v : features[i] = 0
        elif hide == 'hide_project':
            # Hide Project Profile Dimension 
            cols = ['before_pr_project_commits', 'before_pr_project_commits_in_month', 'before_pr_project_prs', 'before_pr_project_prs_in_month', \
                    'before_pr_project_issues_comment', 'before_pr_project_issues_comment_in_month', 'before_pr_project_issues', 'before_pr_project_issues_in_month', \
                    'before_pr_project_issues_comment', 'before_pr_project_issues_comment_in_month', \
                    'before_pr_merge_cnt', 'before_pr_closed_cnt', 'before_pr_merge_ratio']
            for i, v in enumerate(columns.isin(cols)):
                if v : features[i] = 0

        input = desc_title
        if desc_body is not None:
            input += " </s> " + desc_body

        input = input.replace('<nl>', '\n')
        # Remove pulls id \ url \ commits id.
        pattern = re.compile(r'\b\#(\d+)\b')
        input = re.sub(pattern, 'pull_id', input)
        pattern_url = re.compile(r'\bhttps://github.com(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\b')
        input = re.sub(pattern_url, 'pull_id', input)
        input = re.sub(r'\b[0-9a-f]{7,40}\b','commit_id', input)

        if 'code_patch_diff' in self.data.columns:
            diff = info['code_patch_diff'] # " </s> ".join(info['commits'])
        else:
            diff = info['commit_message']

        diff_input = diff.replace('<nl>', '\n')

        if len(input) > self.max_length * 10:
            input = input[:self.max_length * 10]
        if len(diff_input) > self.max_length * 10:
            diff_input = diff_input[:self.max_length * 10]

        if 'BERT' in self.args.model:
            input = self.text_tokenizer.encode_plus(
                input,
                None,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True,
                return_tensors='pt'
            )
            diff_input = self.diff_tokenizer.encode_plus(
                diff_input,
                None,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                return_token_type_ids=True,
                truncation=True,
                return_tensors='pt'
            )
        elif self.args.model in ['VDCNNClassifer', 'LSTM']:
            input = self.text_tokenizer(input)
            diff_input = self.diff_tokenizer(diff_input)
        else:
            input = {}
            diff_input = {}

        # key = ['input_ids','attention_mask','token_type_ids']
        if self.args.model != "baseline":
            for key in input:
                input[key] = torch.cat([input[key], diff_input[key]], 0)
        # if self.args.nominize:
        #     features = [(i-j) / k for i, j, k in zip(features, self.proj_feature_mean[project_name], self.proj_feature_sd[project_name])]
        input['features'] = torch.tensor(features, dtype=torch.float32)
        label = int(info['merge'])
        return pr_url, input, label, "origin", "len"

    def __len__(self):
        return len(self.data)
