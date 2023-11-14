

GPUS=1
CUDA_VISIBLE_DEVICES=0	python train_kfold_with_xgboost.py \
		--model=VDCNNClassifer \
		--fp16=True \
		--batch_size_per_gpu=100 \
		--phase=2 \
		--learning_rate=1e-5 \
		--epochs=3 \
		--max_length=512 \
		--nominize=True\
		--test=True \
		--train_dir=/home/chenkx/github-pr-info-50/results/all_projects_features_Z2_with_code_diff.csv \
		--val_dir=None \
		--test_dir=None \
		--save_model=True\
		--save_dir=./checkpoints/vdcnn-random-10-fold \
		--log_name=logs.txt 
