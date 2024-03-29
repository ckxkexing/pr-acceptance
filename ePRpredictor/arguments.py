# TODO: Remove unnecessary arguments

import argparse
import os

from utils.util import mkdir, str2bool

parser = argparse.ArgumentParser(description="deepPRpredictor Network")
parser.add_argument("--name", type=str, dest="name", default="CodeBERTcnnClassifer")
parser.add_argument("--model", type=str, dest="model", default="baseline")
parser.add_argument(
    "--classifier", type=str, dest="classifier", default="XGBClassifier"
)
parser.add_argument("--glove_emb", type=str2bool, dest="glove_emb", default=False)
parser.add_argument("--desc_backbone", type=str, dest="desc_backbone", default="bert")
parser.add_argument("--diff_backbone", type=str, dest="diff_backbone", default="bert")
parser.add_argument("--dataset_mode", type=str, dest="dataset_mode", default="unique")
parser.add_argument("--dataset", type=str, dest="dataset", default="scrapy_scrapy")
parser.add_argument("--time_series", type=str2bool, dest="time_series", default="False")
parser.add_argument("--input_dir", type=str, dest="input_dir", default=None)
parser.add_argument("--train_dir", type=str, dest="train_dir", default=None)
parser.add_argument("--val_dir", type=str, dest="val_dir", default=None)
parser.add_argument("--test_dir", type=str, dest="test_dir", default=None)
parser.add_argument("--test_on_project", type=str, dest="test_on_project", default=None)
parser.add_argument(
    "--train_test_on_project", type=str, dest="train_test_on_project", default=None
)
parser.add_argument("--label_name", type=str, dest="label_name", default=None)
parser.add_argument("--epochs", type=int, dest="epochs", default=5)
parser.add_argument("--max_length", type=int, dest="max_length", default=100)
parser.add_argument("--seed", type=int, dest="seed", default=47)
parser.add_argument(
    "--batch_size_per_gpu", type=int, dest="batch_size_per_gpu", default=6
)
parser.add_argument("--num_workers", type=int, dest="num_workers", default=4)
parser.add_argument(
    "--local_rank", default=0, type=int, help="parameter used by apex library"
)
parser.add_argument("--learning_rate", default=1e-5, type=float, help="learning rate")
parser.add_argument("--fp16", default=False, type=str2bool, help="fp16 training")
parser.add_argument(
    "--with_lightseq", default=False, type=str2bool, help="with_lightseq training"
)
parser.add_argument("--min_lr", default=1e-5, type=float, help="min learning rate")
parser.add_argument("--log_name", type=str, dest="log_name", default=None)
parser.add_argument(
    "--cl_train",
    default=False,
    type=str2bool,
    help="sim cse constrative learning training",
)
parser.add_argument(
    "--save_model", default=False, type=str2bool, help="save bert + clf model"
)
parser.add_argument("--save_dir", default="", type=str, help="save dir")
parser.add_argument(
    "--test", default="False", type=str2bool, dest="test", help="save dir"
)
parser.add_argument(
    "--gen_test_output",
    default=None,
    type=str,
    dest="gen_test_output",
    help="test output dir",
)
parser.add_argument(
    "--hide",
    default="None",
    type=str,
    help="Hide any part? hide_personal, hide_pr_info or hide_project.",
)
parser.add_argument(
    "--nominize", default=True, type=str2bool, help="check for nominize"
)
parser.add_argument(
    "--phase",
    default=0,
    dest="phase",
    type=int,
    help="2 train phase, 1:train feature layer, 2:textual layer",
)

args = parser.parse_args()
mkdir(args.save_dir)
args.log_name = os.path.join(args.save_dir, args.log_name)
if args.gen_test_output and args.gen_test_output != "None":
    args.gen_test_output = os.path.join(args.save_dir, args.gen_test_output)
