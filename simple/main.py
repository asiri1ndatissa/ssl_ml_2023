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


batch_size = 32
shuffle = True
num_workers = 2

num_classes = 6
d_model = 1086
num_heads = 6
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 2048
dropout = 0.3
num_epochs = 100
lr= 0.001
work_dir = 'work_dirs/exp1'

writer = SummaryWriter(log_dir='logs')

train_dataset = D('/content/drive/MyDrive/asl_kp/train_full.npy',num_classes,100, training=True)
val_dataset = D('/content/drive/MyDrive/asl_kp/val.npy',num_classes,100, training=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# model = TransformerModel(num_classes, d_model, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
model = M(3)
optimizer = AdamW(model.parameters(), lr=lr)
if not os.path.exists(work_dir):
    os.makedirs(work_dir)

def loss_function(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss


for epoch in range(num_epochs):
    model.train()  # Set the model to training mode

    pred_all = 0.
    pred_correct = 0.
    for batch_idx, data in enumerate(train_loader):
        keypoints, labels = data[0], data[1]

        optimizer.zero_grad()  # Clear the gradients
        
        # Forward pass
        outputs = model(keypoints)
        labels_one_hot = F.one_hot(labels, num_classes).float()
        # Compute the loss
        loss = loss_function(outputs, labels_one_hot)
        print('outputs',outputs, labels_one_hot)
        # Backward pass
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        outputs = outputs.argmax(dim=-1)
        pred_all += outputs.shape[0]
        pred_correct += outputs.sum()
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        writer.add_scalar('Loss/Training', accuracy, epoch)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(work_dir, 'model_last.pth'))

        # Log training progress, update metrics, etc.
    acc = pred_correct / pred_all

    # Print accuracy
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Accuracy: {accuracy:.4f} TAcc: {acc:.3f}")

    # print accuracy of training
    # Evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Forward pass
            keypoints, labels = data[0], data[1]
            outputs = model(keypoints)
            
            # Compute the loss and metrics on the validation set
            labels_one_hot = F.one_hot(labels, num_classes).float()

            val_loss = loss_function(outputs, labels_one_hot)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            # Update validation metrics, e.g., accuracy
        print('label predit',labels_one_hot)
        print('label',data[1])
        print(f"validation : {accuracy:.3f}")       
    # Log the training and validation metrics, save checkpoints, etc.
writer.close()
# Training complete

