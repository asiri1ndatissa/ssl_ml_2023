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
class XYZProcessor:
    def __init__(self, npArray):
        self.xyz_array = npArray
        self.newModel = M(3)
        loaded_state_dict = torch.load('/Users/asiriindatissa/src/msc/ssl_ml_2023/work_dirs/exp1/model_lastJuly25th.pth')
        self.newModel.load_state_dict(loaded_state_dict['state_dict'])

    def process_xyz_array(self):
        if hasattr(self, 'xyz_array'):
            print("Loaded XYZ array shape:", self.xyz_array.shape)
            self.newModel.eval()

            print("Loaded XYZ array shape: process_xyz_array")

            val_dataset = D( self.xyz_array,23, 80,training=True)
            val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)

            predicted_texts = ''
            predicted_probs_list = []  # List to store predicted probabilities for each sample

            print('val_loader',val_loader)
            with torch.no_grad():
              for i, data in enumerate(val_loader):
                keypoints, labels = data[0], data[1]
                outputs = self.newModel(keypoints)
                          
                _, predicted = torch.max(outputs, 1)
                predicted_text = [list(sign_map.keys())[list(sign_map.values()).index(pred.item())] for pred in predicted]
                # Compute softmax probabilities
                softmax_probs = F.softmax(outputs, dim=1)
                predicted_probs_list.append(softmax_probs.cpu().numpy())

                # Check if the maximum probability is greater than 0.5
                max_prob, max_class = torch.max(softmax_probs, dim=1)
                print('max_prob',max_prob)
                for prob, text in zip(max_prob, predicted_text):
                    print(prob,text)
                    if prob > 0.5:
                        predicted_texts += text + " "  # Concatenate the predicted text with space
                    else:
                        predicted_texts += "UNKNOWN" + " "
                # predicted_texts += " ".join(predicted_text) + " "  # Concatenate the predicted texts with spaces
                self.predicted_texts = predicted_texts

                accuracy = (predicted == labels).sum().item() / labels.size(0)
                print('accuracy',accuracy, predicted.eq(labels).sum().item(), labels.size(0), predicted_texts)
        else:
            print("XYZ array not loaded. Call 'load_xyz_array()' first.")

    def get_prediction(self):
        return self.predicted_texts