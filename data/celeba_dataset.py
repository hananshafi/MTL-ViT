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


class CelebA(Dataset):

    """CelebA dataset."""

    def __init__(self, root_folder, csv_path, num_tasks=1,task_list=None, transform=None):

        self.root_folder = root_folder
        # self.numpy_arr = numpy_arr
        self.transform = transform
        self.csv_path = csv_path
        self.label_df = pd.read_csv(os.path.join(self.root_folder,self.csv_path))   #pd.read_csv(os.path.join(self.root_folder,"list_attr_celeba.csv"))
        self.num_tasks = num_tasks
        self.task_list = task_list
        df_label = self.label_df[self.task_list]

        df_label = df_label.replace(-1, 0)

        self.task_list_=[]
        for t in list(df_label.columns):
            self.task_list_.append(np.array([np.array(i) for i in df_label[t].tolist()]))


    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #img_name = os.path.join(self.root_dir,
        #                         self.label_annotations.iloc[idx, 0])
        image = Image.open(os.path.join(self.root_folder,"img_align_celeba/img_align_celeba",self.label_df['image_id'].iloc[idx]))
        #image = self.numpy_arr[idx]

        class_labels = [torch.tensor(self.task_list_[i][idx], dtype=torch.float32) for i in range(self.num_tasks)]
        #class_labels = self.label_annotations.iloc[idx, 1:]
        #class_labels = class_labels.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'class_labels': class_labels}

        if self.transform:
            #x = Image.fromarray((self.numpy_arr[idx]*255).astype(np.uint8))
            x = self.transform(image)

        return x,class_labels