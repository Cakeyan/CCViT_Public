#!/usr/bin/env bash

NAME=$1
GPUS=$2
MODE=$3

if [ -z "$3" ]; then
  MODE="print"
fi

BASE_DIR=/data/yanx/CCViT
OUTPUT_DIR="$BASE_DIR/output/$NAME"

echo "$OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

if [ "$MODE" = "nohup" ]; then
    nohup python -m torch.distributed.launch --nproc_per_node=$GPUS $BASE_DIR/run_class_finetuning.py \
        --data_path /data/yanx/ImageNet-1K/ILSVRC/Data/CLS-LOC/train \
        --eval_data_path /data/yanx/ImageNet-1K/ILSVRC/Data/CLS-LOC/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir $OUTPUT_DIR \
        --log_dir $OUTPUT_DIR \
        --model beit_large_patch16_224 \
        --weight_decay 0.05 \
        --finetune $BASE_DIR/output/large_dual_pt300/checkpoint.pth \
        --batch_size 64 \
        --lr 1e-3 \
        --warmup_epochs 5 \
        --epochs 50 \
        --layer_decay 0.8 \
        --drop_path 0.2 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --imagenet_default_mean_and_std   \
        --dist_eval \
        --update_freq 4 \
        --save_ckpt_freq 10 \
        --enable_deepspeed 
        > "$OUTPUT_DIR/print.log" 2>&1 &

elif [ "$MODE" = "print" ]; then
    python -m torch.distributed.launch --nproc_per_node=$GPUS $BASE_DIR/run_class_finetuning.py \
        --data_path /data/yanx/ImageNet-1K/ILSVRC/Data/CLS-LOC/train \
        --eval_data_path /data/yanx/ImageNet-1K/ILSVRC/Data/CLS-LOC/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir $OUTPUT_DIR \
        --log_dir $OUTPUT_DIR \
        --model beit_large_patch16_224 \
        --weight_decay 0.05 \
        --finetune $BASE_DIR/output/large_dual_pt300/checkpoint.pth \
        --batch_size 64 \
        --lr 1e-3 \
        --warmup_epochs 10 \
        --epochs 50 \
        --layer_decay 0.8 \
        --drop_path 0.2 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --imagenet_default_mean_and_std   \
        --dist_eval \
        --update_freq 4 \
        --save_ckpt_freq 10 \
        --enable_deepspeed 
else
    echo "Invalid MODE!"
fi