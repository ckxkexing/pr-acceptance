# 	
#	model choise = [CodeBERTClassifer, xgboost]
#
#	train on other projects
#	test on the project
#	[flutter/flutter , nodejs/node, kubernetes/kubernetes]
#	[angular/angular , mrdoob/three.js] 
#
#	[puppeteer/puppeteer , vercel/next.js , tensorflow/models]
#	[mui-org/material-ui , PanJiaChen/vue-element-admin]


GPUS=1
CUDA_VISIBLE_DEVICES=0	python train_on_csv.py \
		--model=CodeBERTClassifer \
		--classifier=XGBClassifier \
		--fp16=True \
		--desc_backbone="./pretrain/roberta_base" \
		--diff_backbone="./pretrain/microsoft_codebert-base" \
		--batch_size_per_gpu=100 \
		--phase=2 \
		--learning_rate=1e-5 \
		--epochs=2 \
		--max_length=128 \
		--test=True \
		--train_dir=XXXXXX.csv \
		--test_on_project=PanJiaChen/vue-element-admin \
		--val_dir=None \
        --test_dir=None \
		--save_model=True \
		--save_dir=./checkpoints/e-predictor-cross-project \
		--log_name=logs-PanJiaChen-vue-element-admin.txt 