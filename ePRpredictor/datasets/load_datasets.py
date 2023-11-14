from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .dataset_csv import csvDataset, sen_tokenize, standardiseNumericalFeats


def load_data(dir, project_name=None):
    pr_data = pd.read_csv(dir, low_memory=False)

    if project_name:
        pr_data = pr_data[pr_data["pr_url"].str.contains(project_name)].copy()

    for col in [
        "login",
        "github_merge",
        "comment_merge",
        "commit_merge",
        "commit_include_merge",
        "created_at",
        "closed_at",
    ]:
        if col not in list(pr_data):
            continue
        pr_data = pr_data.drop(columns=[col])

    for col in list(pr_data):
        if "in_lifetime" in col:
            pr_data = pr_data.drop(columns=[col])

    for col in ["code_patch_add", "code_patch_del", "commit_message", "title", "body"]:
        if col in list(pr_data):
            pr_data[col] = pr_data[col].fillna("None")
    pr_data = pr_data.fillna(0)

    for col in list(pr_data):
        if col not in [
            "pr_url",
            "merge",
            "code_patch_add",
            "code_patch_del",
            "commit_message",
            "title",
            "body",
        ]:
            pr_data[col] = pr_data[col].astype(np.float32)

    return pr_data


def build_vocab(data, cols):
    counts = Counter()
    for index, row in data.iterrows():
        for col in cols:
            if row[col]:
                input = row[col].replace("<nl>", "\n")
                counts.update(sen_tokenize(input))

    for word in list(counts):
        if counts[word] < 2:
            del counts[word]

    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts.most_common(10000):
        vocab2index[word] = len(words)
        words.append(word)
    return words, vocab2index


def load_datasets(args, logger):
    """ """
    text = (
        "dataset mode : "
        + args.dataset_mode
        + "\n"
        + "train dir : "
        + args.train_dir
        + "\n"
        + "val dir : "
        + args.val_dir
        + "\n"
        + "test dir : "
        + args.test_dir
        + "\n"
    )

    logger.info(text)

    train_data = load_data(args.train_dir)
    val_data = load_data(args.val_dir)
    test_data = load_data(args.test_dir)

    train_data, test_data, val_data = standardiseNumericalFeats(
        train_data, test_data, val_data
    )

    if args.model == "LSTM":
        vocab, vocab2index = build_vocab(train_data, ["title", "body"])
    else:
        vocab, vocab2index = None, None
    train_dataset = csvDataset(
        args,
        X=train_data,
        max_length=args.max_length,
        vocab=vocab,
        vocab2index=vocab2index,
    )
    valid_dataset = csvDataset(
        args,
        X=val_data,
        max_length=args.max_length,
        vocab=vocab,
        vocab2index=vocab2index,
    )
    test_dataset = csvDataset(
        args,
        X=test_data,
        max_length=args.max_length,
        vocab=vocab,
        vocab2index=vocab2index,
    )

    return train_dataset, valid_dataset, test_dataset


def load_datasets_by_project(args, logger):
    data = load_data(args.train_dir)
    test_project_name = args.test_on_project

    train_data = data[~data["pr_url"].str.contains(test_project_name)].copy()
    test_data = data[data["pr_url"].str.contains(test_project_name)].copy()

    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=47)

    train_data, test_data, val_data = standardiseNumericalFeats(
        train_data, test_data, val_data
    )

    if args.model == "LSTM":
        vocab, vocab2index = build_vocab(train_data, ["title", "body"])
    else:
        vocab, vocab2index = None, None

    train_dataset = csvDataset(
        args,
        X=train_data,
        max_length=args.max_length,
        vocab=vocab,
        vocab2index=vocab2index,
    )
    valid_dataset = csvDataset(
        args,
        X=val_data,
        max_length=args.max_length,
        vocab=vocab,
        vocab2index=vocab2index,
    )
    test_dataset = csvDataset(
        args,
        X=test_data,
        max_length=args.max_length,
        vocab=vocab,
        vocab2index=vocab2index,
    )

    return train_dataset, valid_dataset, test_dataset
