import math
import sys
from typing import Iterable
import time
from multiprocessing.dummy import Pool as ThreadPool
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from masking_generator import MaskingGenerator

import utils
import faiss
import numpy as np

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None,
                    index=None, centroids=None, num_training_steps_per_epoch=None, update_freq=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    if args.use_soft_centroid_loss:
        loss_fn = nn.MSELoss()
    elif args.use_dual_loss:
        loss_fn1 = nn.CrossEntropyLoss()
        loss_fn2 = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()


    assert args.patch_size[0] == args.patch_size[1]
    assert args.window_size[0] == args.window_size[1]
    p = args.patch_size[0]
    h = args.window_size[0]
    
    if centroids is not None:
        centroids = centroids.to(device, non_blocking=True)

    for data_iter_step, (batch, extra_info) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        if centroids is not None:
            samples, images, bool_masked_pos, bool_replaced_pos = batch
        else:
            samples, images, bool_masked_pos = batch
        
        samples = samples.to(device, non_blocking=True)

        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        bool_replaced_pos = bool_replaced_pos.to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                if index is not None:
                    images = images.reshape(-1,3,h,p,h,p).permute(0,2,4,1,3,5).reshape(-1,3*p*p)
                    _, input_ids = index.search(images.numpy(), 1)
                    input_ids = torch.from_numpy(input_ids).to(device, non_blocking=True)
                    input_ids = input_ids.reshape(-1,h**2)
                
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)

            if centroids is not None:
                samples = utils.kmeans_mask(bool_replaced_pos, input_ids, centroids, samples)          
                samples = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(samples)
                bool_replaced_pos = bool_replaced_pos.flatten(1).to(torch.bool)
                bool_replaced_pos.logical_or_(bool_masked_pos)
                assert bool_replaced_pos.sum() == bool_masked_pos.sum() + args.num_replaced_patches*args.batch_size

                if args.use_soft_centroid_loss:
                    labels = images.reshape(-1,h*h,3*p*p).to(device, non_blocking=True)
                    mean = labels.mean(dim=-1, keepdim=True)
                    var = labels.var(dim=-1, keepdim=True)
                    labels = (labels - mean) / (var + 1.e-6)**.5
                elif args.use_dual_loss:
                    label1 = input_ids[bool_replaced_pos]
                    label2 = images.reshape(-1,h*h,3*p*p).to(device, non_blocking=True)
                    mean = label2.mean(dim=-1, keepdim=True)
                    var = label2.var(dim=-1, keepdim=True)
                    label2 = (label2 - mean) / (var + 1.e-6)**.5
                    # only on replaced patches
                    label2 = label2[bool_replaced_pos]
                else:
                    labels = input_ids[bool_replaced_pos]
            else:
                labels = input_ids[bool_masked_pos]

        with torch.cuda.amp.autocast(): # enabled=False
            if args.use_soft_centroid_loss:
                outputs = model(samples, bool_masked_pos=bool_masked_pos, bool_replaced_pos=bool_replaced_pos, return_all_tokens=True)
            elif args.use_dual_loss:
                outputs = model(samples, bool_masked_pos=bool_masked_pos, bool_replaced_pos=bool_replaced_pos, return_dual=False)
            elif centroids is not None:
                outputs = model(samples, bool_masked_pos=bool_masked_pos, bool_replaced_pos=bool_replaced_pos)
            else:
                outputs = model(samples, bool_masked_pos=bool_masked_pos)

            if isinstance(outputs, list):
                if args.use_dual_loss:
                    loss_1 = loss_fn1(input=outputs[0], target=label1)
                    loss_2 = loss_fn2(input=outputs[1], target=label2)
                    loss = loss_1 + loss_2 
                else:
                    if args.use_soft_centroid_loss:
                        outputs[0] = torch.softmax(outputs[0], dim=-1)
                        outputs[1] = torch.softmax(outputs[1], dim=-1)
                        outputs[0] = torch.matmul(outputs[0], centroids) 
                        outputs[1] = torch.matmul(outputs[1], centroids)

                    loss_1 = loss_fn(input=outputs[0], target=labels)
                    loss_2 = loss_fn(input=outputs[1], target=labels)
                    loss = loss_1 + loss_2 
            else:
                assert args.use_dual_loss is False
                if args.use_soft_centroid_loss:
                    outputs = torch.softmax(outputs, dim=-1)
                    outputs = torch.matmul(outputs, centroids)
                loss = loss_fn(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
            sys.exit(1)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss /= update_freq
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(data_iter_step + 1) % update_freq == 0)
        if (data_iter_step + 1) % update_freq == 0:
            optimizer.zero_grad()
        
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        
        if isinstance(outputs, list):
            if args.use_dual_loss:
                mlm_acc_1 = (outputs[0].max(-1)[1] == label1).float().mean().item()
                metric_logger.update(mlm_acc_1=mlm_acc_1)
            elif not args.use_soft_centroid_loss:
                mlm_acc_1 = (outputs[0].max(-1)[1] == labels).float().mean().item()
                metric_logger.update(mlm_acc_1=mlm_acc_1)
                mlm_acc_2 = (outputs[1].max(-1)[1] == labels).float().mean().item()
                metric_logger.update(mlm_acc_2=mlm_acc_2)
            metric_logger.update(loss_1=loss_1.item())
            metric_logger.update(loss_2=loss_2.item())

            if log_writer is not None:
                if not args.use_soft_centroid_loss:
                    log_writer.update(mlm_acc_1=mlm_acc_1, head="loss")
                    if not args.use_dual_loss:
                        log_writer.update(mlm_acc_2=mlm_acc_2, head="loss")
                log_writer.update(loss_1=loss_1.item(), head="loss")
                log_writer.update(loss_2=loss_2.item(), head="loss")
        else:
            metric_logger.update(loss=loss.item())
            if not args.use_soft_centroid_loss:
                mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()
                metric_logger.update(mlm_acc=mlm_acc)
            if log_writer is not None:
                log_writer.update(loss=loss.item(), head="loss")
                log_writer.update(mlm_acc=mlm_acc, head="loss")
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            assert False
            lr_scheduler.step_update(start_steps + step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
