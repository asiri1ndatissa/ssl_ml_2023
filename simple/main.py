import argparse
import os
import random
import sys
import logging

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from tModel import TransformerModel
from dataLoading import D
from torch.utils.data import DataLoader


batch_size = 32
shuffle = True
num_workers = 2

num_classes = 6
d_model = 512
num_heads = 8
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 2048
dropout = 0.3
num_epochs = 100
lr= 0.001

train_dataset = D('./drive/MyDrive/asl_data/val.npy',num_classes,80, training=True)
val_dataset = D('./drive/MyDrive/asl_data/val.npy',num_classes,80, training=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


model = TransformerModel(num_classes, d_model, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
optimizer = AdamW(model.parameters(), lr=lr)

def loss_function(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss


for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    
    for batch_idx, (keypoints, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear the gradients
        
        # Forward pass
        outputs = model(keypoints)
        
        # Compute the loss
        loss = loss_function(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        
        # Log training progress, update metrics, etc.
    # Print accuracy
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Accuracy: {accuracy:.4f}")

    # print accuracy of training
    # Evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    
    with torch.no_grad():
        for batch_idx, (keypoints, labels) in enumerate(val_loader):
            # Forward pass
            outputs = model(keypoints)
            
            # Compute the loss and metrics on the validation set
            val_loss = loss_function(outputs, labels)
            
            # Update validation metrics, e.g., accuracy
            
    # Log the training and validation metrics, save checkpoints, etc.

# Training complete

