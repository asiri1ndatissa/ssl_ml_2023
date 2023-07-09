import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from dataset import D, ConcatDataset
from model import M
from adv import EMA, compute_kl_loss, AWP

val_dataset = D('../data/val.npy')
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2)


newModel = M(3)
newModel.cuda()
loaded_state_dict = torch.load('./save_path/exp1/model_final.pth')
model_state_dict = newModel.state_dict()

# print(torch.load('./save_path/exp1/model_final.pth'))
newModel.load_state_dict(model_state_dict, strict=False)

with torch.no_grad():
    output = newModel(input)
    loss = F.cross_entropy(output, label)
