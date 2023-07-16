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



newModel = M(3)
# loaded_state_dict = torch.load('./work_dirs/exp1/model_final.pth')
loaded_state_dict = torch.load('/content/drive/MyDrive/july15thData/model_last.pth')

model_state_dict = newModel.state_dict()

# print(torch.load('./save_path/exp1/model_final.pth'))
newModel.load_state_dict(model_state_dict, strict=False)