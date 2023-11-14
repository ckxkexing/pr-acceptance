# -----------------------------
#	2-phase
#		1. train features linear. phase = 1, lr = 1e-4.
#		2. fix features linear, fine-tune bert. phase = 2, lr = 1e-5.
#		
#		
#		./pretrain/roberta_base
#		./pretrain/microsoft_codebert-base
# -----------------------------

GPUS=1
CUDA_VISIBLE_DEVICES=0	python train_on_csv.py \
		--model=CodeBERTClassifer \
		--fp16=True \
		--desc_backbone="./pretrain/roberta_base" \
		--diff_backbone="./pretrain/microsoft_codebert-base" \
		--batch_size_per_gpu=100 \
		--phase=2 \
		--learning_rate=1e-5 \
		--epochs=2 \
		--max_length=128 \
		--nominize=False \
		--test=True \
		--train_dir=./inputs/train_in_lifetime.csv \
		--val_dir=./inputs/val_in_lifetime.csv \
		--test_dir=./inputs/test_in_lifetime.csv \
		--save_model=True \
		--save_dir=./checkpoints/checkpoints-for-lifetime \
		--gen_test_output=test_predict.csv \
		--log_name=logs.txt