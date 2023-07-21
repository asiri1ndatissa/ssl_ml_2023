import sys
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup
from line_profiler import LineProfiler

from dataLoading import D
from tModel import M

import json

# Load the sign_map.json file
with open('tainFiles/sign_map.json', 'r') as f:
    sign_map = json.load(f)

def process_xyz_array(file_path):
    # Read the array from the file
    xyz_array = np.load(file_path)
    # Process the array as needed
    print("Received XYZ array:", xyz_array.shape)


class XYZProcessor:
    def __init__(self, npArray):
        # self.file_path = 'test_full.npy'
        # self.file_path = file_path
        # self.xyz_array = np.load(self.file_path, allow_pickle=True)
        self.xyz_array = npArray
        self.newModel = M(1)
        loaded_state_dict = torch.load('/Users/asiriindatissa/src/msc/ssl_ml_2023/simple/weights/model_last08.pth')
        self.newModel.load_state_dict(loaded_state_dict['state_dict'])

    def process_xyz_array(self):
        if hasattr(self, 'xyz_array'):
            print("Loaded XYZ array shape:", self.xyz_array.shape)
            # data_list = []
            # data_list.append({'data': self.xyz_array, 'label': 1, 'signer_id': 1})



            self.newModel.eval()

            print("Loaded XYZ array shape: process_xyz_array")

            # val_dataset = D( data_list,23, 100,training=True)
            val_dataset = D( self.xyz_array,23, 100,training=True)
            val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4)

            predicted_texts = ''
            print('val_loader',val_loader)
            with torch.no_grad():
              for i, data in enumerate(val_loader):
                keypoints, labels = data[0], data[1]
                outputs = self.newModel(keypoints)
                          
                _, predicted = torch.max(outputs, 1)
                predicted_text = [list(sign_map.keys())[list(sign_map.values()).index(pred.item())] for pred in predicted]
                predicted_texts += " ".join(predicted_text) + " "  # Concatenate the predicted texts with spaces
                self.predicted_texts = predicted_texts

                accuracy = (predicted == labels).sum().item() / labels.size(0)
                print('accuracy',accuracy, predicted.eq(labels).sum().item(), labels.size(0), predicted_texts)
        else:
            print("XYZ array not loaded. Call 'load_xyz_array()' first.")

    def get_prediction(self):
        return self.predicted_texts