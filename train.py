import argparse
import os
import random
import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset import D, ConcatDataset
from model import M
from adv import EMA, compute_kl_loss, AWP, compute_js_divergence

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--work_dir', type=str, default='work_dirs/exp1')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--pretrained', type=str, default="")
    args = parser.parse_args()
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.cuda.device_count())]
    output = torch.cat(tensors_gather, dim=0)
    return output

args = parse_args()
setup_seed(args.seed)

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
print_freq = args.print_freq
work_dir = args.work_dir
warmup_ratio = args.warmup_ratio
nlayers = args.nlayers

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

logger = logging.getLogger('log')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

fh = logging.FileHandler(work_dir + '/log.txt')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

if args.debug:
    train_dataset = D('./data/val.npy', training=True)
else:
    train_dataset = D('./data/train_full.npy', training=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

if args.debug:
    val_dataset = D('./data/val.npy')
else:
    val_dataset = D('./data/val.npy')
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)

model = M(nlayers)
model.cuda()

ema = EMA(model, 0.9998)
ema.register()

opt = AdamW(model.parameters(), lr=lr)
stepsize = len(train_dataset) // batch_size + 1
total_steps = stepsize * epochs
scheduler = get_linear_schedule_with_warmup(opt, int(total_steps * warmup_ratio), total_steps)
awp = AWP(model=model, optimizer=opt, adv_lr=3e-5,adv_eps=1e-2, start_epoch=1)

def train_epoch(epoch):
    model.train()
    torch.cuda.synchronize()
    train_loss = 0
    for i, data in enumerate(train_loader):
        input, label = data[0], data[1]
        input, label = input.cuda(), label.cuda()

        output = model(input)
        res_softmax = F.softmax(output, dim=1)
        loss = F.cross_entropy(output, label, label_smoothing=0.5)
        label_onehot = F.one_hot(label, num_classes=23).float()
        loss_poly = 1. - (label_onehot * res_softmax).sum(dim=1).mean() # poly loss
        loss += loss_poly
        loss_kl = compute_js_divergence(res_softmax, label_onehot) # rdrop loss
        loss += loss_kl
        loss.backward()

        awp.attack_backward(epoch) # AWP attach, and then re-compute the loss again
        output = model(input)
        res_softmax = F.softmax(output, dim=1)
        loss = F.cross_entropy(output, label, label_smoothing=0.5)
        label_onehot = F.one_hot(label, num_classes=23).float()
        loss_poly = 1. - (label_onehot * res_softmax).sum(dim=1).mean() # poly loss
        loss += loss_poly
        loss_kl = compute_js_divergence(res_softmax, label_onehot) # rdrop loss
        loss += loss_kl
        loss.backward()
        awp.restore()

        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0) # clip the gradient  
        opt.step()
        ema.update()
        scheduler.step()
        opt.zero_grad()


        # if (i + 1) % print_freq == 0:
        #     train_loss /= print_freq
        #     logger.info(f"Epoch [{epoch}/{epochs}] Batch [{i+1}/{stepsize}]\tLoss: {train_loss:.4f}")
        #     train_loss = 0
    logger.info(f"Epoch [{epoch}/{epochs}] Batch [{i+1}/{stepsize}]\tLoss: {train_loss:.4f}")
    model.eval()
    ema.apply_shadow()

def eval_epoch(epoch):
    torch.cuda.synchronize()
    pred_all = 0.
    pred_correct = 0.
    for i, data in enumerate(val_loader):
        input, label = data[0], data[1]
        input, label = input.cuda(), label.cuda()
        with torch.no_grad():
            output = model(input)
        output = output.argmax(dim=-1)
        tmp = (output == label).float()
        pred_all += output.shape[0]
        pred_correct += output.sum()
    acc = pred_correct / pred_all
    logger.info(f"Epoch [{epoch}/{epochs}] Validation Accuracy: {acc:.2f}%")

def main():
    logger.info("Start training...")
    for epoch in range(epochs):
        train_epoch(epoch)
        eval_epoch(epoch)
        torch.cuda.synchronize()
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
        }, os.path.join(work_dir, 'model_last.pth'))

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict(),
    }, os.path.join(work_dir, 'model_final.pth'))

if __name__ == '__main__':
    main()