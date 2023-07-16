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


batch_size = 20
shuffle = True
num_workers = 2

num_classes = 23
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

train_dataset = D('/content/ssl_ml_2023/data/train_full.npy',num_classes,100, training=True)
val_dataset = D('/content/ssl_ml_2023/data/val.npy',num_classes,100, training=True)

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
    print('epoch',epoch)
    model.train()  # Set the model to training mode

    total = 0.
    correct = 0.
    epoch_loss = 0
    for batch_idx, data in enumerate(train_loader):

        keypoints, labels = data[0], data[1]

        optimizer.zero_grad()  # Clear the gradients
        
        # Forward pass
        outputs = model(keypoints)
        labels_one_hot = F.one_hot(labels, num_classes).float()
        # Compute the loss
        loss = loss_function(outputs, labels_one_hot)
        epoch_loss += loss.item()
        # Backward pass
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        outputs = outputs.argmax(dim=-1)
        # pred_all += outputs.shape[0]
        # pred_correct += outputs.sum()
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        writer.add_scalar('Loss/Training', accuracy, epoch)
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(work_dir, 'model_last.pth'))
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log training progress, update metrics, etc.
    accu=100.*correct/total
    avg_loss = epoch_loss / len(train_loader)

    # Print accuracy
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Accuracy: {accuracy:.4f} loss: {avg_loss:.3f} TAccuracy: {accu:.3f}")

    # print accuracy of training
    # Evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode

    val_total = 0.
    val_correct = 0.
    val_epoch_loss = 0
    
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
            val_epoch_loss += val_loss.item()
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

        val_accu=100.*val_correct/val_total
        val_avg_loss = val_epoch_loss / len(train_loader)

            # Update validation metrics, e.g., accuracy
        print(f"validation : {val_accu:.3f} loss : {val_epoch_loss:.3f} avg_loss : {val_avg_loss:.3f}")       
    # Log the training and validation metrics, save checkpoints, etc.
writer.close()
# Training complete

