# 	
#	model choise = [CodeBERTClassifer, xgboost]
#
#	train on other projects
#	test on the project
#	[flutter/flutter , nodejs/node, kubernetes/kubernetes]
#	[angular/angular , mrdoob/three.js] 
#	[puppeteer/puppeteer , vercel/next.js , tensorflow/models]
#	[mui-org/material-ui , laravel/laravel]


GPUS=1
CUDA_VISIBLE_DEVICES=0	python train_kfold_with_xgboost.py \
		--model=CodeBERTClassifer \
		--fp16=True \
		--desc_backbone="./pretrain/roberta_base" \
		--diff_backbone="./pretrain/microsoft_codebert-base"\
		--batch_size_per_gpu=100 \
		--phase=2 \
		--learning_rate=1e-5 \
		--epochs=2 \
		--max_length=128 \
		--time_series=True \
		--nominize=True\
		--test=True \
		--train_dir=/home/chenkx/github-pr-info-50/results/all_projects_features_Z2_in_lifetime_sorted.csv \
		--train_test_on_project=laravel/laravel \
        --val_dir=None \
        --test_dir=None \
		--save_model=True\
		--save_dir=./checkpoints/e-predictor-timeseries-single-project \
		--log_name=logs-laravel-laravel.txt 