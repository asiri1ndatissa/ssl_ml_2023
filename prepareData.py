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

LANDMARK_FILES_DIR = "/content/drive/MyDrive/july8thData/keyPoints"
TRAIN_FILE = "/content/drive/MyDrive/july8thData/train_file.csv"
label_map = json.load(open("/content/drive/MyDrive/july8thData/sign_map.json", "r"))

# create directory for datasets
os.system('mkdir data')

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

MAX_LEN = 256

def convert_and_save_validate_data(val=False):
    df = pd.read_csv(TRAIN_FILE)
    df['label'] = df['sign'].map(label_map)

    if val:
        df = df[df.signer_id.isin([2])]
    else:
        df = df[~df.signer_id.isin([2])]

    data_list = []

    indexs = np.arange(df.shape[0], dtype=int)
    np.random.seed(1008600)
    np.random.shuffle(indexs)

    for i, (path, label, signer_id) in tqdm(enumerate(df[['path', 'label', 'signer_id']].values[indexs]), total=df.shape[0]):
        data, label, signer_id = convert_row(path, label, signer_id)
        if i == 0:
            print(data.shape, label, data, signer_id)
        data_list.append({'data': data, 'label': label, 'signer_id': signer_id})

    if val:
        print('Val')
        np.save(f"./data/val.npy", np.array(data_list))
    else:
        np.save(f"./data/train_full.npy", np.array(data_list))

print('test')
convert_and_save_validate_data(True)