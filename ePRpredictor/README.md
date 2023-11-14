# E-PRedictor

### Prepare
- mkdir `pretrain`,download Huggingfaces's microsoft_codebert-base、roberta_base pretrain package and put in it.
- unzip dataset.zip, update all `cmds` files `input dir path`. 
### RQ1
#### Random 10 fold
|  methods   | cmd  |
|  ----  | ----  |
| VDCNN     | cmds/train_vdcnn_10.sh |
| E-PRedicor| cmds/train_bert_xgboost_10.sh |
| others    | cmds/train_baseline_10.sh|
#### time-aware 10 fold
use Function `solve_gen_time_series_dataset` in  
|  methods   | cmd  |
|  ----  | ----  |
| VDCNN     | cmds/time_series_train_vdcnn_10.sh |
| E-PRedicor| cmds/time_series_train_bert_xgboost_10.sh |
| others    | cmds/time_series_train_baseline_10.sh|

### RQ2-hide some features
use `cmds/train_bert_xgboost_10_hide.sh`
you can set hide_dim with:
  - hide_personal
  - hide_info
  - hide_project
  - hide_desc
  - hide_diff


### RQ3-cross project setting
you have to gen cross project data by using `solve_gen_cross_dataset` in `gen_json_data.py`

then change `train_dir` and `val_dir` in `cmds/train.sh` or `cmds/train_xgboost.sh`



### RQ4-single project time-aware-setting
you have to gen single project data by using `solve_gen_one_project` in `gen_json_data.py`

then change `train_dir` and `val_dir` in `cmds/train.sh` or `cmds/train_xgboost.sh`