# Centroid-centered Modeling for Efficient Vision Transformer Pre-training

The official codebase of [CCViT](https://arxiv.org/abs/2303.04664).

## Setup

```
pip install -r requirements.txt
```

Install [apex](https://github.com/NVIDIA/apex) for mixed-precision training

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Also, you need to download ImageNet-1K and ADE20K dataset. 

## Training Centroids via Faiss

```bash
python train_faiss_kmeans.py --img_root_path /path/of/imagenet --ngpu 8 --k 8192 --n_iter 20 --sample 50 --exp_name train_50000_8192 --output_path ./faiss
```

This will generate index file and centroids file.

## Pretraining on ImageNet-1k

```bash
CUDA_VISIBLE_DEVICES=5 python -m torch.distributed.launch --nproc_per_node=1  --master_port 29876 run_ccvit_pretraining.py \
        --data_set image_folder \
        --data_path /path/of/imagenet \
        --output_dir /path/output \
        --log_dir /path/log \
        --model ccvit_base_patch16_224_8k_vocab_cls_pt \
        --shared_lm_head False \
        --early_layers 9 \
        --head_layers 2 \
        --num_mask_patches 75 \
        --num_replaced_patches 20 \
        --second_input_size 224 \
        --second_interpolation bicubic \
        --min_crop_scale 0.2 \
        --kmeans_index /path/faiss/index \
        --centroids_path /path/faiss/centroids \
        --batch_size 128 \
        --lr 1.5e-3 \
        --warmup_epochs 10 \
        --clip_grad 3.0 \
        --drop_path 0. \
        --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8 \
        --epochs 300 \
        --save_ckpt_freq 20 \
        --update_freq 2 \
        --use_dual_loss True
```

## Fine-tuning on ImageNet-1k (Image Classification)

```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 29876 run_class_finetuning.py \
        --data_path /path/of/imagenet/train \
        --eval_data_path  /path/of/imagenet/val \
        --nb_classes 1000 \
        --data_set image_folder \
        --output_dir /path/output \
        --log_dir /path/log \
        --model ccvit_base_patch16_224 \
        --weight_decay 0.05 \
        --batch_size 128 \
        --lr 2e-3 \
        --update_freq 1 \
        --warmup_epochs 20 \
        --epochs 100 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --mixup 0.8 \
        --cutmix 1.0 \
        --imagenet_default_mean_and_std \
        --dist_eval \
        --save_ckpt_freq 20 \
        --finetune /path/of/pretrained/model
```

## Fine-tuning on ADE20K 

Install the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library and some required packages. Follow the guide in [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md) to prepare the ADE20k dataset.

```bash
pip install mmcv-full==1.3.0 mmsegmentation==0.11.0
pip install scipy timm==0.3.2
```

```bash
cd semantic_segmentation; 
bash tools/dist_train.sh \
    configs/ccvit/upernet/upernet_ccvit_base_12_512_slide_160k_ade20k.py 8 \
    --work-dir output/ --seed 0  --deterministic \
    --options model.pretrained=/path/of/pretrained/model
```

## Acknowledgement

Heavily rely on codebase from [beit2](https://github.com/microsoft/unilm/tree/master/beit2)

## Citation

If you find this repository useful, please consider citing our work:

```
@misc{ccvit,
      title={Centroid-centered Modeling for Efficient Vision Transformer Pre-training}, 
      author={Xin Yan and Zuchao Li and Lefei Zhang},
      year={2023},
      eprint={2303.04664},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2303.04664}, 
}
```
