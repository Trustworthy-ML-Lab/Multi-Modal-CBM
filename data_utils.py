import os
import torch
from torchvision import datasets, transforms, models
from loguru import logger
import numpy as np
import clip
from torch.utils.data import Dataset
import json
from typing import List, Dict, Tuple, Any


DATASET_FOLDER = os.environ.get("DATASET_FOLDER", "/YOUR_PATH")

DATASET_ROOTS = {
    "imagenet_train": "/data/imagenet/train",
    "imagenet_val": "/space/MMCBM/Multimodal-CBM/imagenet/val",
    "cub_train": "/data/CUB/train",
    "cub_val": "/space/MMCBM/Multimodal-CBM/cub/test"
}

LABEL_FILES = {"places365":"data/classes/categories_places365_clean.txt",
               "imagenet":"data/classes/imagenet_classes.txt",
               "cifar10":"data/classes/cifar10_classes.txt",
               "cifar100":"data/classes/cifar100_classes.txt",
               "cub":"data/classes/cub_classes.txt",
               "food101":"data/classes/food101_classes.txt",
               "oxfordpets":"data/classes/oxfordpets_classes.txt",
               "flowers102":"data/classes/flowers102_classes.txt",
               "dtd":"data/classes/dtd_classes.txt",
               "eurosat":"data/classes/eurosat_classes.txt"
               }


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser(DATASET_FOLDER), download=True, train=True, transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser(DATASET_FOLDER), download=True, train=False, transform=preprocess)

    elif dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser(DATASET_FOLDER), download=True, train=True, transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser(DATASET_FOLDER), download=True, train=False, transform=preprocess)
        
    elif dataset_name == "oxfordpets_train":
        data = datasets.OxfordIIITPet(root=os.path.expanduser(DATASET_FOLDER), split="trainval", target_types="category", download=True,
                                       transform=preprocess)
        
    elif dataset_name == "oxfordpets_val":
        data = datasets.OxfordIIITPet(root=os.path.expanduser(DATASET_FOLDER), split="test", target_types="category", download=True,
                                       transform=preprocess)
        
    elif dataset_name == "flowers102_train":
        data = torch.utils.data.ConcatDataset([datasets.Flowers102(root=os.path.expanduser(DATASET_FOLDER), split="train", download=True, transform=preprocess),
                                                datasets.Flowers102(root=os.path.expanduser(DATASET_FOLDER), split="val", download=True, transform=preprocess)])

    elif dataset_name == "flowers102_val":
        data = datasets.Flowers102(root=os.path.expanduser(DATASET_FOLDER), split="test", download=True, transform=preprocess)
    
    elif dataset_name == "food101_train":
        data = datasets.Food101(root=os.path.expanduser(DATASET_FOLDER), split="train", download=True, transform=preprocess)

    elif dataset_name == "food101_val":
        data = datasets.Food101(root=os.path.expanduser(DATASET_FOLDER), split="test", download=True, transform=preprocess)

    elif dataset_name == "dtd_train":
        data = torch.utils.data.ConcatDataset([datasets.DTD(root=os.path.expanduser(DATASET_FOLDER), split="train", download=True, transform=preprocess),
                                                datasets.DTD(root=os.path.expanduser(DATASET_FOLDER), split="val", download=True, transform=preprocess)])

    elif dataset_name == "dtd_val":
        data = datasets.DTD(root=os.path.expanduser(DATASET_FOLDER), split="test", download=True, transform=preprocess)


    elif dataset_name == "eurosat_train":
        dataset = datasets.EuroSAT(root=os.path.expanduser(DATASET_FOLDER), download=True, transform=preprocess, target_transform=None)
        data, _ = torch.utils.data.random_split(dataset, [10000, len(dataset) - 10000], generator=torch.Generator().manual_seed(42))

    elif dataset_name == "eurosat_val":
        dataset = datasets.EuroSAT(root=os.path.expanduser(DATASET_FOLDER), download=True, transform=preprocess, target_transform=None)
        _, data = torch.utils.data.random_split(dataset, [len(dataset) - 5000, 5000], generator=torch.Generator().manual_seed(42))
        
    elif dataset_name == "places365_train":
        try:
            data = datasets.Places365(root=os.path.expanduser(DATASET_FOLDER), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser(DATASET_FOLDER), split='train-standard', small=True, download=False,
                                   transform=preprocess)
            
    elif dataset_name == "places365_val":
        try:
            data = datasets.Places365(root=os.path.expanduser(DATASET_FOLDER), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            data = datasets.Places365(root=os.path.expanduser(DATASET_FOLDER), split='val', small=True, download=False,
                                   transform=preprocess)
            
    elif dataset_name == "food101_train":
        data = datasets.Food101(root=os.path.expanduser(DATASET_FOLDER), split="train", download=True, transform=preprocess)

    elif dataset_name == "food101_val":
        data = datasets.Food101(root=os.path.expanduser(DATASET_FOLDER), split="test", download=True, transform=preprocess)
        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    return data

def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    if hasattr(pil_data, 'targets'):
        return pil_data.targets
    else:
        targets = [pil_data[i][1] for i in range(len(pil_data))]
        return targets

def get_target_model(target_name, device):
    
    if target_name.startswith("CLIP_"):
        target_name = target_name[5:]
        target_name = target_name.rsplit("-", 1)
        target_name = "/".join(target_name)
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()

    return target_model, preprocess


class CBMDataset(Dataset):
    def __init__(self, backbone_act_train_img, target_train_img, train_targets):
        self.data = backbone_act_train_img
        self.target_od = target_train_img
        self.target = train_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target_od[idx], self.target[idx]