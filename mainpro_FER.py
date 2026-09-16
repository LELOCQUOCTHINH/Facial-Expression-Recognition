# -*- coding: utf-8 -*-
"""Train Fer2013 with PyTorch (NPZ dataset + modern PyTorch, Windows-safe)."""

from __future__ import print_function

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import transforms as transforms  # transforms của repo gốc
import utils
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from fer import FER2013
from models import *

# ---------------------- TOP-LEVEL helpers (pickle-safe) ----------------------
def tencrop_stack(crops):
    """
    TenCrop trả về list[PIL.Image]. Hàm này chuyển mỗi crop -> Tensor và stack lại
    thành Tensor shape (10, C, H, W). Phải để ở top-level để DataLoader (spawn)
    có thể pickle được trên Windows.
    """
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


# --------------------------------- ARGS --------------------------------------
parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='VGG19',
                    choices=['VGG19', 'Resnet18'], help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FER2013',
                    help='dataset name (for checkpoint dir)')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PublicTest_acc = 0.0
best_PublicTest_acc_epoch = 0
best_PrivateTest_acc = 0.0
best_PrivateTest_acc_epoch = 0
start_epoch = 0

learning_rate_decay_start = 20
learning_rate_decay_every = 5
learning_rate_decay_rate = 0.9

cut_size = 44
total_epoch = 64

path = os.path.join(f"{opt.dataset}_{opt.model}")


# ------------------------------- DATA LOADERS --------------------------------
def build_loaders():
    print('==> Preparing data..')

    transform_train = transforms.Compose([
        transforms.RandomCrop(cut_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        transforms.Lambda(tencrop_stack),  # dùng hàm top-level (pickle-safe)
    ])

    trainset = FER2013(split='Training',    transform=transform_train)
    publicset = FER2013(split='PublicTest', transform=transform_test)
    privateset = FER2013(split='PrivateTest', transform=transform_test)

    # không còn h5py nên có thể dùng num_workers>0 an toàn
    num_workers = 4
    pin = bool(use_cuda)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.bs, shuffle=True,
        num_workers=num_workers, pin_memory=pin
    )
    PublicTestloader = torch.utils.data.DataLoader(
        publicset, batch_size=opt.bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    PrivateTestloader = torch.utils.data.DataLoader(
        privateset, batch_size=opt.bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin
    )
    return trainloader, PublicTestloader, PrivateTestloader


# --------------------------------- MODEL -------------------------------------
def build_model():
    print('==> Building model..')
    if opt.model == 'VGG19':
        net = VGG('VGG19')
    else:
        net = ResNet18()

    if use_cuda:
        net = net.cuda()
        cudnn.benchmark = True
    return net


def resume_if_needed(net):
    global best_PublicTest_acc, best_PrivateTest_acc
    global best_PublicTest_acc_epoch, best_PrivateTest_acc_epoch, start_epoch

    if not opt.resume:
        return net

    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    ckpt_path = os.path.join(path, 'PrivateTest_model.t7')
    checkpoint = torch.load(ckpt_path, map_location='cuda' if use_cuda else 'cpu')

    net.load_state_dict(checkpoint['net'])
    best_PublicTest_acc = float(checkpoint.get('best_PublicTest_acc', checkpoint.get('acc', 0.0)))
    best_PrivateTest_acc = float(checkpoint.get('best_PrivateTest_acc', 0.0))
    best_PublicTest_acc_epoch = int(checkpoint.get('best_PublicTest_acc_epoch', 0))
    best_PrivateTest_acc_epoch = int(checkpoint.get('best_PrivateTest_acc_epoch', checkpoint.get('epoch', 0)))
    start_epoch = best_PrivateTest_acc_epoch + 1
    print(f"Resumed at epoch {start_epoch}, best_PrivateTest_acc={best_PrivateTest_acc:.3f}")
    return net


# ------------------------------ TRAIN / EVAL ---------------------------------
def set_lr_if_needed(optimizer, epoch):
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)
    else:
        current_lr = opt.lr
    print(f'learning_rate: {current_lr}')
    return current_lr


def train_one_epoch(net, criterion, optimizer, trainloader, epoch):
    print(f'\nEpoch: {epoch}')
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0

    set_lr_if_needed(optimizer, epoch)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.detach(), 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        utils.progress_bar(
            batch_idx, len(trainloader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total)
        ) 

    Train_acc = 100. * correct / total
    return Train_acc


@torch.no_grad()
def eval_public(net, criterion, PublicTestloader, epoch):
    global best_PublicTest_acc, best_PublicTest_acc_epoch
    net.eval()
    PublicTest_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
        # inputs: (B, 10, C, H, W)
        bs, ncrops, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)  # (B*10, C, H, W)
        if use_cuda:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over 10 crops
        loss = criterion(outputs_avg, targets)
        PublicTest_loss += loss.item()

        _, predicted = torch.max(outputs_avg, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        utils.progress_bar(
            batch_idx, len(PublicTestloader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total)
        )

    PublicTest_acc = 100. * correct / total
    if PublicTest_acc > best_PublicTest_acc:
        print('Saving..')
        print("best_PublicTest_acc: %0.3f" % PublicTest_acc)
        state = {
            'net': net.state_dict(),
            'acc': PublicTest_acc,
            'epoch': epoch,
        }
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, 'PublicTest_model.t7'))
        best_PublicTest_acc = PublicTest_acc
        best_PublicTest_acc_epoch = epoch
    return PublicTest_acc


@torch.no_grad()
def eval_private(net, criterion, PrivateTestloader, epoch):
    global best_PrivateTest_acc, best_PrivateTest_acc_epoch
    global best_PublicTest_acc, best_PublicTest_acc_epoch

    net.eval()
    PrivateTest_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        bs, ncrops, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)

        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()

        _, predicted = torch.max(outputs_avg, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        utils.progress_bar(
            batch_idx, len(PrivateTestloader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total)
        )

    PrivateTest_acc = 100. * correct / total

    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict(),
            'best_PublicTest_acc': best_PublicTest_acc,
            'best_PrivateTest_acc': PrivateTest_acc,
            'best_PublicTest_acc_epoch': best_PublicTest_acc_epoch,
            'best_PrivateTest_acc_epoch': epoch,
        }
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, 'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch
    return PrivateTest_acc


# ----------------------------------- MAIN ------------------------------------
def main():
    trainloader, PublicTestloader, PrivateTestloader = build_loaders()
    net = build_model()
    net = resume_if_needed(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=5e-4)


    for epoch in range(start_epoch, total_epoch):
        _ = train_one_epoch(net, criterion, optimizer, trainloader, epoch)
        # _ = eval_public(net, criterion, PublicTestloader, epoch)
        _ = eval_private(net, criterion, PrivateTestloader, epoch)

    print("best_PublicTest_acc: %0.3f" % best_PublicTest_acc)
    print("best_PublicTest_acc_epoch: %d" % best_PublicTest_acc_epoch)
    print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
    print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)


if __name__ == '__main__':
    import multiprocessing as mp
    mp.freeze_support()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
