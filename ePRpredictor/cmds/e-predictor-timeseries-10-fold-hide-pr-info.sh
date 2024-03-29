
GPUS=1
CUDA_VISIBLE_DEVICES=2	python train_kfold_with_xgboost.py \
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
		--hide=hide_pr_info \
		--nominize=True\
		--test=True \
		--train_dir=./inputs/all_projects_features_Z2_in_lifetime_sorted.csv \
		--val_dir=None \
        --test_dir=None \
		--save_model=True\
		--save_dir=./checkpoints/e-predictor-timeseries-10-fold-hide-pr-info \
		--log_name=logs-2.txt 
