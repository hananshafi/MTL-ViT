import torch.nn as nn
import torch
import requests
import torchvision.transforms as transforms
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import timm
from torchsummary import summary
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torchmetrics.classification.accuracy import Accuracy
import torchvision.models as models
from sklearn.utils import class_weight
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial
# from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from data.celeba_dataset import CelebA
from data.cifar10_dataset import MTLCIFAR10
from config import ModelParams
from model.vision_transformer import MTLVisionTransformer

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

os.environ['TORCH_HOME'] = 'env-path' #setting the environment variable



def get_class_weights(df, col_name):
  class_weights = class_weight.compute_class_weight(class_weight ='balanced',
                                                 classes = np.unique(df[col_name].to_list()),
                                                y = df[col_name].to_list())
  return list(class_weights)



def get_combined_acc(acc_list):
  acc_sum=0
  for i in acc_list:
    acc_sum += i
  return acc_sum

class CustomVITLitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        #self.model= ViT( image_size=224, patch_size=model_params.patch_size, num_classes=model_params.n_classes, dim=model_params.projection_dim, depth=model_params.transformer_layers,
        #        heads=model_params.num_heads, mlp_dim=model_params.projection_dim*2, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.1, emb_dropout = 0.1)
        self.model = MTLVisionTransformer(pretrained=True)
        #self.model=MTLVisionTransformer(pretrained=True)
        self.criterion = [nn.CrossEntropyLoss(weight=class_weights[i]) for i in range(model_params.num_tasks)]
        self.criterion_test = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.val_accuracy =  Accuracy()
        self.test_accuracy =  Accuracy()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        logits = self.model(x)
        return logits


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, labels = batch
        labels = [torch.reshape(labels[i].long(),(-1,)) for i in range(model_params.num_tasks)]
        logits = self.forward(x)
        loss =[self.criterion[i](logits[i], labels[i].long()) for i in range(model_params.num_tasks)]

        preds = [torch.argmax(logits[i], dim=1) for i in range(model_params.num_tasks)]
        acc_all = [self.accuracy(preds[index],labels[index]) for index in range(model_params.num_tasks)]
        self.log("train_loss", get_combined_acc(loss))
        #self.log("train_acc", get_combined_acc(acc_all))
        self.log("train_acc",torch.mean(torch.tensor(acc_all)))
        for ii,t_loss in enumerate(loss):
            self.log(f"train/loss-{ii}", t_loss)
        for ii,t_acc in enumerate(acc_all):
            self.log(f"train/acc-{ii}", t_acc)

        return {"loss": get_combined_acc(loss), "preds": preds, "targets": labels}

    def validation_step(self, batch, batch_idx: int):
        x, targets = batch
        targets = [torch.reshape(targets[i].long(),(-1,)) for i in range(model_params.num_tasks)]
        logits = self.forward(x)

        loss =[self.criterion[i](logits[i], targets[i].long()) for i in range(model_params.num_tasks)]

        preds = [torch.argmax(logits[i], dim=1) for i in range(model_params.num_tasks)]
        # log val metrics
#         preds_cpu = preds.cpu()
#         targets_cpu = targets.cpu()

        acc = [self.val_accuracy(preds[i], targets[i]) for i in range(model_params.num_tasks)]
        self.log("val_acc",torch.mean(torch.tensor(acc)))
        self.log("val_loss",torch.mean(torch.tensor(loss)))
        #self.log("val_acc",get_combined_acc(acc))
        #f1  = f1_score(targets_cpu,preds_cpu,average="weighted")
        for ii,v_loss in enumerate(loss):
            self.log(f"val/loss-{ii}", v_loss)
        for ii,v_acc in enumerate(acc):
            self.log(f"val/acc-{ii}", v_acc,on_step=False, on_epoch=True, prog_bar=True)
#         self.log({f"val_{index+1}_acc":acc  for index,acc in enumerate(acc)})
#         self.log({f"val_{i+1}_loss": loss[i] for index in range(model_params.num_tasks)})
        return {"loss": get_combined_acc(loss), "preds": preds, "targets": targets}

    def test_step(self, batch, batch_idx: int):
        x, targets = batch
        logits = self.forward(x)
        preds = [torch.argmax(logits[i], dim=1) for i in range(model_params.num_tasks)]
        loss =[self.criterion_test(logits[i], targets[i].long()) for i in range(model_params.num_tasks)]
        # log val metrics
#         preds_cpu = preds.cpu()
#         targets_cpu = targets.cpu()

        acc = [self.test_accuracy(preds[i], targets[i].int()) for i in range(model_params.num_tasks)]
        comb_test_loss = get_combined_acc(loss)
        self.log("test_acc",torch.mean(torch.tensor(acc)))
        self.log("test_loss",torch.mean(torch.tensor(loss)))
        #f1  = f1_score(targets_cpu,preds_cpu,average="weighted")
        for index,t_acc in enumerate(acc):
            self.log(f"test/acc-{index+1}", t_acc)
        # self.log({f"test_{index+1}_acc":acc  for index,acc in enumerate(acc)})
        for index,t_loss in enumerate(loss):
            self.log(f"test/loss-{index+1}", t_loss)
        #self.log({f"test_{i+1}_loss": loss[i] for i in range(model_params.num_tasks)})
        return {"loss": comb_test_loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=model_params.lr)
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":

	model_params = ModelParams()
	Tasks = model_params.celeba_tasks

	normalize =  transforms.Normalize(mean = np.array([0.485, 0.456, 0.406]),
                std = np.array([0.229, 0.224, 0.225]))
	train_dataset = CelebA(root_folder="./CelebA_data",csv_path="train.csv",num_tasks=model_params.num_tasks,task_list=Tasks,
                       transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize]))

	val_dataset = CelebA(root_folder="./CelebA_data",csv_path="validation.csv",num_tasks=model_params.num_tasks, task_list=Tasks,
                     transform=transforms.Compose([
					 transforms.Resize((224,224)),transforms.ToTensor(),normalize]))

	test_dataset = CelebA(root_folder="./CelebA_data",csv_path="test.csv",num_tasks=model_params.num_tasks, task_list=Tasks,
                      transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normalize]))


	train_loader = DataLoader(
        train_dataset,
        batch_size=model_params.BS,
        shuffle=True,
        num_workers=10,pin_memory=True)
	val_loader = DataLoader(
        val_dataset,
        batch_size=model_params.BS,
        shuffle=False,
        num_workers=5,pin_memory=True)
	test_loader = DataLoader(
        test_dataset,
        batch_size=model_params.BS,
        shuffle=False,
        num_workers=2)


	label_df = pd.read_csv(os.path.join("./CelebA_data","list_attr_celeba.csv"))
	class_weights = [torch.Tensor(get_class_weights(label_df, Tasks[i])).to(device) for i in range(model_params.num_tasks)]

	if not os.path.exists("./saved_model"):
		os.mkdir("./saved_model")

	early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=10, verbose=False, mode="min")
	checkpoint_callback = ModelCheckpoint(
		dirpath="./saved_model_vit_imagenet_pre",
		filename="deit-tiny-iter3-imagenet_pre-all-tasks-checkpoint",
		save_top_k=2,
		verbose=True,
		monitor="val_loss",
		mode="min")
	trainer = pl.Trainer(max_epochs=100,gpus=[0],auto_lr_find=True,callbacks=[early_stop_callback,checkpoint_callback],
						 progress_bar_refresh_rate=30)
	model = CustomVITLitModule()
	print(model_params.num_tasks)
	trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
	trainer.test(model,test_loader)
