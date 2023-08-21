import os
from sklearn.metrics import f1_score,precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from tModel import M
from dataLoading import DataProcess
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from confusionMatrix import  classificationReport
from AdaLoss import AdaLoss


writer = SummaryWriter("runs/sslr_ada_loss_dp_0.2_aug_01_120_dim_M_RRelu_1_affine_up_model")
# writer.close()

batch_size = 20
shuffle = True
num_workers = 0
num_classes = 23

num_heads = 6
num_encoder_layers = 2
num_decoder_layers = 2
num_epochs = 100
lr= 0.001
work_dir = 'work_dirs/exp1'

load_train = np.load('/Users/asiriindatissa/src/msc/ssl_ml_2023/data/train_fullSSL.npy',allow_pickle=True)
load_val = np.load('/Users/asiriindatissa/src/msc/ssl_ml_2023/data/valSSL.npy',allow_pickle=True)


train_dataset = DataProcess(load_train,num_classes,80, training=True)
val_dataset = DataProcess(load_val,num_classes,80, training=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

num_training_steps = len(train_loader) * num_epochs
num_warmup_steps = int(0.1 * num_training_steps)

model = M(3)
optimizer = AdamW(model.parameters(), lr=lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
dummy_input = torch.randn(batch_size, 979, requires_grad=True)  # Adjust the tensor dimensions as necessary

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

def loss_function(outputs, targets):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    return loss

adaloss_criterion = AdaLoss(num_classes, beta=1.0)


for epoch in range(num_epochs):
    print('epoch',epoch)
    model.train()  # Set the model to training mode

    total = 0.
    correct = 0.
    epoch_loss = 0
    lRate = 0

    for batch_idx, data in enumerate(train_loader):

        keypoints, labels = data[0], data[1]
        optimizer.zero_grad()  # Clear the gradients
        # Forward pass
        outputs = model(keypoints)


        labels_one_hot = F.one_hot(labels, num_classes).float()

        # Compute the loss
        # loss = loss_function(outputs, labels_one_hot)
        loss = adaloss_criterion(outputs, labels)

        epoch_loss += loss.item()
        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        _, predicted = torch.max(outputs, 1)
        outputs = outputs.argmax(dim=-1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Log training progress, update metrics, etc.
    if(epoch % 10 ==0 ):
      torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(work_dir, 'model_last.pth'))
    if epoch == 1:
        writer.add_graph(model, keypoints)

    accu=100.*correct/total
    avg_loss = epoch_loss / len(train_loader)
    writer.add_scalar('training loss', avg_loss, epoch)
    writer.add_scalar('training accuracy', accu, epoch)
    

    for param_group in optimizer.param_groups:
        lRate = param_group['lr']
        print(f"Learning Rate: {param_group['lr']:.8f}")
    # Print accuracy
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Accuracy: {accuracy:.4f} loss: {avg_loss:.3f} TAccuracy: {accu:.3f} Learning Rate: {lRate:.8f}")

    # print accuracy of training
    # Evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode

    val_total = 0.
    val_correct = 0.
    val_epoch_loss = 0
    
    labels_pr = []
    pred_pr = []
    y_pred = [] # save predction
    y_true = [] # save ground truth
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            # Forward pass
            keypoints, labels = data[0], data[1]
            outputs = model(keypoints)
            
            # Compute the loss and metrics on the validation set
            labels_one_hot = F.one_hot(labels, num_classes).float()
            val_loss = adaloss_criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == labels).sum().item() / labels.size(0)
            val_epoch_loss += val_loss.item()
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
            # PR curve
            class_predictions = [F.softmax(output, dim =0) for output in outputs]

            pred_pr.append(class_predictions)
            labels_pr.append(predicted)
            y_pred.append(predicted)
            y_true.append(labels)
           

        pred_pr = torch.cat([torch.stack(batch) for batch in pred_pr])
        labels_pr = torch.cat(labels_pr)
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        y_pred = y_pred.numpy()
        y_true = y_true.numpy()
        print('predicted',y_pred.shape)
        print('labels',y_true.shape)
        print('shape of labels_pr, pred_pr',labels_pr.shape, pred_pr.shape)

        val_accu=100.*val_correct/val_total
        val_avg_loss = val_epoch_loss / len(train_loader)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='micro')  # or 'macro' depending on your preference
        recall = recall_score(y_true, y_pred, average='micro')  # or 'macro' depending on your preference

        writer.add_scalar('validation loss', val_avg_loss, epoch)
        writer.add_scalar('validation accuracy', val_accu, epoch)
        writer.add_scalar('precision',precision,epoch)
        writer.add_scalar('recall',recall,epoch)
        classificationReport(y_pred, y_true,epoch)
        print(f"validation : {val_accu:.3f} loss : {val_epoch_loss:.3f} avg_loss : {val_avg_loss:.3f}")       

        classes = range(num_classes)
        for i in classes:
            labels_i = labels_pr == i
            preds_i = pred_pr[:, i]
            writer.add_pr_curve(str(i), labels_i,preds_i, global_step =0)
            writer.close()

# Training complete

