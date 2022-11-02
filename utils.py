'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict
import quantization

def enable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child, quantization.quantizer.Quantizer):
            child.ptq = True
        else:
            enable_calibrate(child)
    return module

def disable_calibrate(module):
    for name, child in module.named_children():
        if isinstance(child, quantization.quantizer.Quantizer):
            child.ptq = False
        else:
            disable_calibrate(child)
    return module

def disable_soft_targets(module):
    for name, child in module.named_children():
        if isinstance(child, quantization.quantizer.Quantizer):
            child.soft_targets = False
        else:
            disable_soft_targets(child)
    return module
class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


# def calibrate_params(module):
#     for name, child in module.named_children():
#         if isinstance(child, quantization.quantizer.AdaRoundQuantizer):
#             child.soft_targets = False
#         else:
#             disable_soft_targets(child)
#     return module

def calibrate_adaround(module, adaround_iter, b_start, b_end, warmup, trainloader, device):

    opt_params = []
    for name, child in module.named_modules():
        if isinstance(child, quantization.quantizer.AdaRoundQuantizer):
            # print('child.alpha: ', child.alpha)
            opt_params += [child.alpha]
            # print(opt_params)
    optimizer = torch.optim.Adam(opt_params)
    scheduler = None

    temp_decay = LinearTempDecay(adaround_iter, rel_start_decay=warmup,
                                          start_b=b_start, end_b=b_end)
    for j in range(adaround_iter):    
        b = temp_decay(j)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if batch_idx == 10: break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = module(inputs)
            round_loss = 0
            for name, child in module.named_modules():
                if isinstance(child, quantization.quantizer.AdaRoundQuantizer):
                    round_vals = child.get_soft_targets()
                    round_loss += (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            optimizer.zero_grad()
            round_loss.backward()
            optimizer.step()   
    for name, child in module.named_modules():
        if isinstance(child, quantization.quantizer.AdaRoundQuantizer):
            child.soft_targets = False

def inplace_linear(linear, ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq):
    new_layer = quantization.QLinear(ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq,
                linear.in_features, linear.out_features,
                True if linear.bias is not None else False)
    new_layer.weight = linear.weight
    print(new_layer.weight.device)
    if linear.bias is not None:
        new_layer.bias = linear.bias
    return new_layer

def inplace_conv(conv, ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq):
    new_layer = quantization.QConv2d(ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq,
                conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                conv.padding, conv.dilation, conv.groups,
                True if conv.bias is not None else False)
    new_layer.weight = conv.weight
    if conv.bias is not None:
        new_layer.bias = conv.bias
    
    return new_layer

def inplace_conv_bn(conv, bn, total_steps, ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq):
    new_layer = quantization.QConv2dBn(ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq, total_steps, bn,
                conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                conv.padding, conv.dilation, conv.groups,
                True if conv.bias is not None else False)
    new_layer.weight = conv.weight
    if conv.bias is not None:
        new_layer.bias = conv.bias
    
    return new_layer

def inplace_quantize_layers(module, total_steps, ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq):
    last_conv_flag = 0
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child, (nn.modules.batchnorm._BatchNorm)):
            if last_conv is None:
                continue
            fused_qconv = inplace_conv_bn(last_conv, child, total_steps, ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq)
            module._modules[last_conv_name] = fused_qconv
            module._modules[name] = nn.Identity()
            last_conv = None
            last_conv_flag = 0

        if last_conv_flag == 1:
            qconv = inplace_conv(last_conv, ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq)
            module._modules[last_conv_name] = qconv
            last_conv = None
            last_conv_flag = 0

        if isinstance(child, nn.Conv2d):
                last_conv = child
                last_conv_name = name
                last_conv_flag = 1

        if isinstance(child, nn.Linear):
                qlinear = inplace_linear(child, ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq)
                module._modules[name] = qlinear
        else:
            inplace_quantize_layers(child, total_steps, ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq)
    return module

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)

term_width=80

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()

def add_module_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict