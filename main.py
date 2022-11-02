'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import quantization
import os
import argparse

from models import *
from utils import progress_bar, inplace_quantize_layers, enable_calibrate, disable_calibrate, calibrate_adaround, add_module_dict


parser = argparse.ArgumentParser(description='PyTorch MNIST QUANTIZE Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--type', choices=['fp32','PTQ','QAT'], help='choose train fp32, PTQ or QAT')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dorefa', '-d', action='store_true', help='use dorefa to quantizate')
parser.add_argument('--Histogram', action='store_true', help='use HistogramObserver to quantizate')
parser.add_argument('--omse', action='store_true', help='use omse to quantizate')
parser.add_argument('--lsq', action='store_true', help='use lsq to quantizate')
parser.add_argument('--bias_correction', action='store_true', help='use bias_correction to quantizate')
parser.add_argument('--level', default='L', choices=['L','C'],  help='per_channel or per_tensor')
parser.add_argument('--path', default='./checkpoint/',  help='model saved path')

parser.add_argument('--adaround', action='store_true', help='use adaround to quantizate')
parser.add_argument('--adaround-iter', default=1000, type=int)
parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')



args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_epochs = 20

# Data
print('==> Preparing data..')

trainset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG_s')
net = net.to(device)

# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

if args.type == "PTQ" or args.lsq:
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    
    new_state_dict = add_module_dict(checkpoint['net'])
    # net.load_state_dict(checkpoint['net'])
    net.load_state_dict(new_state_dict)


if args.type == "PTQ" or args.type == "QAT":
    net = inplace_quantize_layers(net, len(trainloader) * train_epochs, ptq = True if args.type == "PTQ" else False,
                             dorefa = args.dorefa, Histogram = args.Histogram, level = args.level, omse = args.omse,
                             adaround = args.adaround, bias_correction = args.bias_correction, lsq = args.lsq)
    net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def calibrate():
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx == 10: break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

def calibrate_ada(net):
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_idx == 10: break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return net
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        if args.type == 'fp32':
            torch.save(state, './checkpoint/ckpt.pth')
        else:
            torch.save(state, './checkpoint/ckpt_q.pth')
        best_acc = acc

for epoch in range(start_epoch, start_epoch + train_epochs):
    if epoch == start_epoch:
        enable_calibrate(net)
        calibrate()
        disable_calibrate(net)
        if args.adaround:
            calibrate_adaround(net, args.adaround_iter, args.b_start, args.b_end, args.warmup, trainloader, device)
        test(epoch)
        if args.type == "PTQ":
            break
    else:
        train(epoch)
        test(epoch)
        scheduler.step()
