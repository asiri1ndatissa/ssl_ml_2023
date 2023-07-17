import os
import gc

import json
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings(action='ignore')

# create directory for datasets
os.system('mkdir data')
TEST_PATH = '/content/drive/MyDrive/july15thData/keyPoints/කලු/Black7.csv'

class FeatureGen(nn.Module):
    def __init__(self):
        super(FeatureGen, self).__init__()
        pass
    
    def forward(self, x):
        LEN = x.shape[0]

        return x.reshape(LEN, -1)
    
feature_converter = FeatureGen()

ROWS_PER_FRAME = 543
def load_relevant_data_subset(csv_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_csv(csv_path, usecols=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


def convert_row(path, label, signer_id):
    x = load_relevant_data_subset(os.path.join("./", path))
    return feature_converter(torch.tensor(x)).cpu().numpy(), label, signer_id

def convert_and_save_test_data():
    data_list = []
    data, label, signer_id = convert_row(TEST_PATH, 20, 3)
    print(data.shape, label, data, signer_id)
    data_list.append({'data': data, 'label': label, 'signer_id': signer_id})
    np.save(f"./data/test_full.npy", np.array(data_list))


convert_and_save_test_data()