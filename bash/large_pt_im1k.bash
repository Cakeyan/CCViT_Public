#!/usr/bin/env bash

NAME=$1
GPUS=$2
MODE=$3

if [ -z "$3" ]; then
  MODE="print"
fi

BASE_DIR=/data/yanx/lab/CCViT
OUTPUT_DIR="$BASE_DIR/output/$NAME"

echo "$OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

if [ "$MODE" = "nohup" ]; then
    nohup python -m torch.distributed.launch --nproc_per_node=$GPUS $BASE_DIR/run_beitv2_pretraining.py \
        --data_set image_folder \
        --data_path /data/yanx/lab/dataset/ILSVRC/Data/CLS-LOC/train \
        --output_dir $OUTPUT_DIR \
        --log_dir $OUTPUT_DIR \
        --model beit_large_patch16_224_8k_vocab_cls_pt \
        --shared_lm_head False \
        --early_layers 21 \
        --head_layers 2 \
        --num_mask_patches 75 \
        --num_replaced_patches 20 \
        --second_input_size 224 \
        --second_interpolation bicubic \
        --min_crop_scale 0.2 \
        --tokenizer kmeans \
        --kmeans_index "$BASE_DIR/data/train_50000_8192/train_50000_8192_index.faiss" \
        --centroids_path "$BASE_DIR/data/train_50000_8192/train_50000_8192_centroids.npy" \
        --batch_size 32 \
        --lr 1.5e-3 \
        --warmup_epochs 10 \
        --clip_grad 3.0 \
        --drop_path 0. \
        --layer_scale_init_value 1e-5 \
        --imagenet_default_mean_and_std \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 300 \
        --save_ckpt_freq 20 \
        --update_freq 16 \
        --use_dual_loss True \
        > "$OUTPUT_DIR/print.log" 2>&1 &

elif [ "$MODE" = "print" ]; then
    python -m torch.distributed.launch --nproc_per_node=$GPUS $BASE_DIR/run_beitv2_pretraining.py \
        --data_set image_folder \
        --data_path /data/yanx/lab/dataset/ILSVRC/Data/CLS-LOC/train \
        --output_dir $OUTPUT_DIR \
        --log_dir $OUTPUT_DIR \
        --model beit_large_patch16_224_8k_vocab_cls_pt \
        --shared_lm_head False \
        --early_layers 21 \
        --head_layers 2 \
        --num_mask_patches 75 \
        --num_replaced_patches 20 \
        --second_input_size 224 \
        --second_interpolation bicubic \
        --min_crop_scale 0.2 \
        --tokenizer kmeans \
        --kmeans_index "$BASE_DIR/data/train_50000_8192/train_50000_8192_index.faiss" \
        --centroids_path "$BASE_DIR/data/train_50000_8192/train_50000_8192_centroids.npy" \
        --batch_size 32 \
        --lr 1.5e-3 \
        --warmup_epochs 10 \
        --clip_grad 3.0 \
        --drop_path 0. \
        --layer_scale_init_value 1e-5 \
        --imagenet_default_mean_and_std \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 300 \
        --save_ckpt_freq 20 \
        --update_freq 16 \
        --use_dual_loss True
else
    echo "Invalid MODE!"
fi