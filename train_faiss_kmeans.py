from ast import Raise
import numpy as np
import time
import faiss
import sys
from torchvision import transforms as pth_transforms
import torch
from PIL import Image
import os
import torchvision
import os
from os import path
import argparse
import random


def train_kmeans(x, args):
    "Runs kmeans on one or several GPUs"
    k = args.k
    ngpu = args.ngpu
    n_iter = args.n_iter
    # n_redo = args.n_redo
    exp_name = args.exp_name
    output_path = args.output_path
    memory = args.memory * 1024 * 1024 * 1024

    try:
        os.mkdir(output_path)
        os.mkdir(path.join(output_path, exp_name))
    except:
        pass

    output_path = path.join(output_path, exp_name)

    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    clus.seed = 1
    clus.verbose = True
    clus.niter = n_iter

    # otherwise the kmeans implementation sub-samples the training set
    clus.max_points_per_centroid = 10000000

    res = [faiss.StandardGpuResources() for i in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        res[0].setTempMemory(memory)
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                   for i in range(ngpu)]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    clus.train(x, index)

    D, I = index.search(x, 1)

    centroids = faiss.vector_float_to_array(clus.centroids)

    stats = clus.iteration_stats
    obj = np.array([stats.at(i).obj for i in range(stats.size())])
    print("final objective: %.4g" % obj[-1])

    faiss.write_index(faiss.index_gpu_to_cpu(index), path.join(output_path,exp_name+'_index.faiss'))
    np.save(path.join(output_path,exp_name+'_centroids.npy'), centroids.reshape(k, d)) 
    np.savetxt(path.join(output_path,exp_name+'_I.txt'),I.reshape(-1,196), fmt='%d')

    return


def main(args):
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor()
    ])
    array = []

    # notice that k=8192 so that patches should less than 8.192M, which is only 41.7K photos
    sample = args.sample
    img_root_path = args.img_root_path
    num_patches = args.num_patches
    p = args.patch_size
    h = int(num_patches ** 0.5)
    # XXX
    assert h == 14
    
    sample_flag = -1
    if sample != 1:
        for i in os.listdir(img_root_path):
            img_path = path.join(img_root_path, i)
            sample_flag += 1
            print("Folder No."+str(sample_flag)+" now. Processing in: ", img_path)
            n_all_file = len(os.listdir(img_path))
            np.random.seed(1)
            rand_array = np.random.choice(n_all_file, sample, replace=False)
            for idx in range(rand_array.shape[0]):
                with open(path.join(img_path,os.listdir(img_path)[rand_array[idx]]), 'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB')
                    img = transform(img) # 0~1
                    t = img.reshape(3,h,p,h,p).permute(1,3,0,2,4).reshape(num_patches,-1)
                    array.append(t)
                
    data = torch.cat(array)
    print(data.shape, data.max(), data.min(), data.dtype)

    print("run")
    t0 = time.time()

    train_kmeans(data.numpy(), args)

    t1 = time.time()
    print("total runtime: %.3f s" % (t1 - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get code for K-means from faiss.')
    parser.add_argument('--img_root_path', default='/path/of/your/dataset', type=str, help="dataset path")
    parser.add_argument('--ngpu', default=1, type=int, help="num of gpu used, use with CUDA_VISIBLE_DEVICES to specify the GPU in the server")
    parser.add_argument('--k', default=8192, type=int, help="num of clusters in K-means")
    parser.add_argument('--n_iter', default=20, type=int, help="num of ieration times in K-means")
    parser.add_argument('--n_redo', default=1, type=int, help="num of re-do times in K-means")
    parser.add_argument('--memory', default=20, type=int, help="how many Gigbites memory used in GPU")
    parser.add_argument('--sample', default=50, type=int, help="select sample images in each sub-folder, we select 50 images of 1000 subfolders, so it is 50000.")
    parser.add_argument('--exp_name', default='train_50000_8192', type=str, help="'dataset_imgs_tokens', name of the experiment, used in saving files")
    parser.add_argument('--output_path', default='', type=str, help="output path of the experiment, used in saving files")
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--num_patches', default=196, type=int)
    args = parser.parse_args()

    main(args)
