

GPUS=1
CUDA_VISIBLE_DEVICES=0	python train_on_csv.py \
		--model=CodeBERTClassifer \
		--fp16=True \
		--desc_backbone="./pretrain/roberta_base" \
		--diff_backbone="./pretrain/roberta_base" \
		--batch_size_per_gpu=100 \
		--phase=2 \
		--learning_rate=1e-5 \
		--epochs=3 \
		--max_length=128 \
		--nominize=False \
		--test=True \
		--train_dir=./inputs/train.csv \
		--val_dir=./inputs/val.csv \
        --test_dir=./inputs/test.csv \
		--save_model=True \
		--save_dir=./checkpoints/checkpoints-phase2 \
		--log_name=logs.txt 