import argparse
import os
import torch
import random

from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic, _pil_interp
from timm.data import create_transform, ImageDataset 

from masking_generator import MaskingGenerator, RandomMaskingGenerator
from dataset_folder import ImageFolder
import numpy as np

class DataAugmentationForCCViT(object):
    def __init__(self, args):
        self.use_replaced_masking = args.centroids_path != ''
        self.num_replaced_patches = args.num_replaced_patches
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size, scale=(args.min_crop_scale, 1.0),
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.replaced_transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=args.input_size, second_size=args.second_input_size, scale=(args.min_crop_scale, 1.0),
                interpolation=args.train_interpolation, second_interpolation=args.second_interpolation,
            ),
        ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),])                                             

        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        if self.use_replaced_masking:  
            masked_position = self.masked_position_generator()
            false_indices = np.where(masked_position.flatten() == False)[0]
            np.random.shuffle(false_indices)
            bool_replaced_pos = np.zeros_like(masked_position).astype(np.int).flatten()
            index = false_indices[:self.num_replaced_patches]
            bool_replaced_pos[index] = 1
            bool_replaced_pos = bool_replaced_pos.reshape(masked_position.shape)

            for_patches, for_visual_tokens = self.replaced_transform_train(image)
            return \
                self.visual_token_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
                masked_position, bool_replaced_pos
        else:
            for_patches, for_visual_tokens = self.common_transform(image)
            return \
                self.patch_transform(for_patches), self.visual_token_transform(for_visual_tokens), \
                self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForCCViT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        # repr += "  Replaced position generator = %s,\n" % str(self.replaced_position_generator)
        repr += ")"
        return repr

def build_ccvit_pretraining_dataset(args):
    transform = DataAugmentationForCCViT(args)
    print("Data Aug = %s" % str(transform))
    
    return ImageFolder(args.data_path, transform=transform)

############################################### Dataset and Transforms for Ft #########################################################

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        try:
            index_file = args.image_folder_class_index_file
        except:
            index_file = None
        dataset = ImageFolder(root, transform=transform, index_file=index_file)
        try:
            nb_classes = args.nb_classes
            assert len(dataset.class_to_idx) == nb_classes
        except:
            nb_classes = 1000
            assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    # assert nb_classes == args.nb_classes
    print("Number of the class = %d" % nb_classes)
    return dataset, nb_classes


def build_transform(is_train, args):
    try:
        resize_im = args.input_size > 32 and not args.val_reconstructed
    except:
        resize_im = args.input_size > 32
    print(resize_im)
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        try:
            args.crop_pct = args.crop_pct
        except:
            args.crop_pct = 224 / 256
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    # t.append(transforms.ColorJitter(0.4, 0.4, 0.4))
    t.append(transforms.Normalize(mean, std))

    try:
        if args.val_reconstructed:
            t.append(transforms.GaussianBlur(kernel_size=41, sigma=4))
    except:
        pass

    print(transforms.Compose(t))
    return transforms.Compose(t)
