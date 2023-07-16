import argparse
import os
import random
import sys
import logging
import math
from torch.utils.tensorboard import SummaryWriter


from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from tModel import M
from dataLoading import D
from torch.utils.data import DataLoader

def loss_function(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss

newModel = M(1)
loaded_state_dict = torch.load('/content/drive/MyDrive/july15thData/model_last08.pth')
newModel.load_state_dict(loaded_state_dict['state_dict'])

newModel.eval()

val_dataset = D('/content/ssl_ml_2023/data/val.npy',23, 100,training=True)
val_loader = DataLoader(val_dataset, batch_size=20, num_workers=2)

val_total = 0.
val_correct = 0.
val_epoch_loss = 0
with torch.no_grad():
  for i, data in enumerate(val_loader):
    keypoints, labels = data[0], data[1]
    outputs = newModel(keypoints)
              
    # Compute the loss and metrics on the validation set
    labels_one_hot = F.one_hot(labels, 23).float()

    val_loss = loss_function(outputs, labels_one_hot)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == labels).sum().item() / labels.size(0)
    val_epoch_loss += val_loss.item()
    val_total += labels.size(0)
    val_correct += predicted.eq(labels).sum().item()
  val_accu=100.*val_correct/val_total
  val_avg_loss = val_epoch_loss / len(val_loader)
print(f"validation : {val_accu:.3f} loss : {val_epoch_loss:.3f}")  