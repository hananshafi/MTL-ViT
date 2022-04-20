import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms, utils
import torch.nn as nn
from PIL import Image
import os
import pandas as pd
import numpy as np
import math
from functools import partial
import torchvision


class MTLCIFAR10(Dataset):

    """CIFAR10 dataset."""

    def __init__(self,train=True,num_tasks=10,task_list=None, mode="mtl", transform=None):

        self.dataset = torchvision.datasets.CIFAR10(root='./data', train=train,
                                   download=True,transform = transform)
        self.task_dict = {"airplane":0,
                        "automobile":1,
                        "bird" :2,
                        "cat":3,
                        "deer" :4,
                        "dog" :5,
                        "frog" :6,
                        "horse":7,
                        "ship" :8,
                        "truck":9
            }
        self.num_tasks = num_tasks
        self.task_list = task_list
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.dataset[idx]
        image = data[0]
        label = data[1]
        if model_params.num_tasks > 1:
            class_label =[1 if i == label else 0 for i in range(model_params.num_tasks)]
        else:
            target = self.task_dict[self.task_list[0]]
            class_label = [1 if label == target else 0]
        
        class_label = [torch.tensor(class_label[i], dtype=torch.float32) for i in range(model_params.num_tasks)]
        return image,class_label