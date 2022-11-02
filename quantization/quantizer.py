
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .observer import OmseObserver
class Quantizer(nn.Module):
    def __init__(self, bit, observer, ptq, sign=False):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
    
    def update_qparams(self, tensor):
        raise NotImplementedError
    
    def forward(self, tensor):
        if self.training or self.ptq:
            self.observer(tensor)
            self.update_qparams(tensor)
        quant_tensor = (torch.round(tensor / self.scale) - tensor / self.scale).detach() + tensor / self.scale + self.zero_point
        fake_quant_tensor = (quant_tensor - self.zero_point) * self.scale

        return fake_quant_tensor
  
class AsymmetricQuantizer(Quantizer):
    def __init__(self, bit, observer, ptq, sign=False):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq

        if self.observer.level == "L":
            self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
            self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))
        elif self.observer.level == "C":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
        elif self.observer.level == "FC":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1), dtype=torch.float32),
            )
        # self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
        # self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))

        self.register_buffer("quant_min",
                              torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32),
                            )

        self.register_buffer("quant_max",
                              torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32),
                            )
        self.register_buffer("eps", 
                              torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32)
                            )
    
    def update_qparams(self, inputs):

        if isinstance(self.observer, OmseObserver):
            best_score = 1e+10
            for i in range(90):
                new_max = self.observer.max_val * (1.0 - (i * 0.01))
                new_min = self.observer.min_val * (1.0 - (i * 0.01))
                new_scale = (new_max - new_min) / float(self.quant_max - self.quant_min)
                new_scale.clamp_(self.eps)
                new_zero_point = self.quant_min - torch.round(new_min / new_scale)
                new_zero_point.clamp_(self.quant_min, self.quant_max)
                inputs_q = ((inputs / new_scale + new_zero_point).round().clamp(
                    self.quant_min, self.quant_max) - new_zero_point) * new_scale
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(inputs, inputs_q, p=2.0, reduction='all')
                if score < best_score:
                    best_score = score
                    self.observer.max_val = new_max
                    self.observer.min_val = new_min
                    scale = new_scale
                    zero_point = new_zero_point

        else:    
            scale = (self.observer.max_val - self.observer.min_val) / (self.quant_max - self.quant_min)
            zero_point = (torch.round(self.quant_min - self.observer.min_val / scale) - (self.quant_min - self.observer.min_val / scale)).detach() + \
                        (self.quant_min - self.observer.min_val / scale)

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)


class AdaRoundQuantizer(Quantizer):

    def __init__(self, bit, observer, ptq, sign=False, round_mode='learned_hard_sigmoid'):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.observer = observer
        self.ptq = ptq
        self.round_mode = round_mode
        self.alpha = None
        self.ada_init = None
        self.soft_targets = True
        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3

        
        # self.init_alpha(x=weight_tensor.clone())
        if self.observer.level == "L":
            self.register_buffer("scale", torch.ones((1), dtype=torch.float32))
            self.register_buffer("zero_point", torch.zeros((1), dtype=torch.float32))
        elif self.observer.level == "C":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1, 1, 1), dtype=torch.float32),
            )
        elif self.observer.level == "FC":
            self.register_buffer(
                "scale",
                torch.ones((self.observer.out_channels, 1), dtype=torch.float32),
            )
            self.register_buffer(
                "zero_point",
                torch.zeros((self.observer.out_channels, 1), dtype=torch.float32),
            )
        self.register_buffer("quant_min",
                              torch.tensor((-(1 << (self.bit - 1))), dtype=torch.float32),
                            )

        self.register_buffer("quant_max",
                              torch.tensor(((1 << (self.bit - 1)) - 1), dtype=torch.float32),
                            )
        self.register_buffer("eps", 
                              torch.tensor((torch.finfo(torch.float32).eps), dtype=torch.float32)
                            )

    
    def update_qparams(self, inputs):
        if isinstance(self.observer, OmseObserver):
            best_score = 1e+10
            for i in range(90):
                new_max = self.observer.max_val * (1.0 - (i * 0.01))
                new_min = self.observer.min_val * (1.0 - (i * 0.01))
                new_scale = (new_max - new_min) / float(self.quant_max - self.quant_min)
                new_scale.clamp_(self.eps)
                new_zero_point = self.quant_min - torch.round(new_min / new_scale)
                new_zero_point.clamp_(self.quant_min, self.quant_max)
                inputs_q = ((inputs / new_scale + new_zero_point).round().clamp(
                    self.quant_min, self.quant_max) - new_zero_point) * new_scale
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(inputs, inputs_q, p=2.0, reduction='all')
                if score < best_score:
                    best_score = score
                    self.observer.max_val = new_max
                    self.observer.min_val = new_min
                    scale = new_scale
                    zero_point = new_zero_point

        else:    
            scale = (self.observer.max_val - self.observer.min_val) / (self.quant_max - self.quant_min)
            zero_point = (torch.round(self.quant_min - self.observer.min_val / scale) - (self.quant_min - self.observer.min_val / scale)).detach() + \
                        (self.quant_min - self.observer.min_val / scale)

        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)

    def forward(self, tensor):
        if self.training or self.ptq:
            self.observer(tensor)
            self.update_qparams(tensor)

        # print('self.ada_init', self.ada_init)
        if not self.ada_init:
            self.init_alpha(tensor.clone())
            self.ada_init = True
        # quant_tensor = (torch.round(tensor / self.scale) - tensor / self.scale).detach() + tensor / self.scale + self.zero_point
        # fake_quant_tensor = (quant_tensor - self.zero_point) * self.scale


        quant_tensor = self.quant(tensor)
        fake_quant_tensor = self.dequantize(quant_tensor)

        return fake_quant_tensor

    def quant(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        # range_shape = self.get_reshape_range(inputs)
        # scale = scale.reshape(range_shape)
        # zero_point = zero_point.reshape(range_shape)

        if self.round_mode == 'nearest':
            x_int = torch.round(inputs / scale)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(inputs / scale)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(inputs / scale)
            rest = (inputs / scale) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            print('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(inputs / scale)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                print('test test test')
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')

        outputs = x_int + zero_point
        outputs = outputs.round().clamp(self.quant_min,
                                        self.quant_max)
        return outputs

    def dequantize(self, inputs, scale=None, zero_point=None):
        if scale is None:
            scale = self.scale
        if zero_point is None:
            zero_point = self.zero_point
        # range_shape = self.get_reshape_range(inputs)
        # scale = scale.reshape(range_shape)
        # zero_point = zero_point.reshape(range_shape)
        outputs = (inputs - zero_point) * scale
        return outputs
    
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):

        scale = self.scale
        # range_shape = self.get_reshape_range(x)
        # scale = scale.reshape(range_shape)
        x_floor = torch.floor(x / scale)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = (x / scale) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError



class LsqQuantizer(nn.Module):
    def __init__(self, bit, level, ptq, sign=False, all_positive=False, symmetric=False):
        super().__init__()

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = - 2 ** (bit - 1)
                self.thd_pos = 2 ** (bit - 1) - 1
        if level != 'L':
            self.per_channel = True
        else:
            self.per_channel = False

        self.s = nn.Parameter(torch.ones(1))

    def init_from(self, x, *args, **kwargs):
        if self.per_channel:
            self.s = nn.Parameter(
                x.detach().abs().mean(dim=list(range(1, x.dim())), keepdim=True) * 2 / (self.thd_pos ** 0.5))
        else:
            self.s = nn.Parameter(x.detach().abs().mean() * 2 / (self.thd_pos ** 0.5))

    def forward(self, x):
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        s_scale = self.grad_scale(self.s, s_grad_scale)
        print(self.s)

        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)
        x = x * s_scale
        return x
    
    def grad_scale(self, x, scale):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad


    def round_pass(self, x):
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad



class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# A(特征)量化
class dorefa_ActivationQuantizer(nn.Module):
    def __init__(self, a_bits):
        super(dorefa_ActivationQuantizer, self).__init__()
        self.a_bits = a_bits

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input):
        if self.a_bits == 32:
            output = input
        elif self.a_bits == 1:
            print("！Binary quantization is not supported ！")
            assert self.a_bits != 1
        else:
            output = torch.clamp(input * 0.1, 0, 1)  # 特征A截断前先进行缩放（* 0.1），以减小截断误差
            scale = 1 / float(2 ** self.a_bits - 1)  # scale
            output = self.round(output / scale) * scale  # 量化/反量化
        return output


# W(权重)量化
class dorefa_WeightQuantizer(nn.Module):
    def __init__(self, w_bits):
        super(dorefa_WeightQuantizer, self).__init__()
        self.w_bits = w_bits

    # 取整(ste)
    def round(self, input):
        output = Round.apply(input)
        return output

    # 量化/反量化
    def forward(self, input):
        if self.w_bits == 32:
            output = input
        elif self.w_bits == 1:
            print("！Binary quantization is not supported ！")
            assert self.w_bits != 1
        else:
            output = torch.tanh(input)
            max_w = torch.max(torch.abs(output)).detach()
            output = output / 2 / max_w + 0.5  # 归一化-[0,1]
            scale = 1 / float(2 ** self.w_bits - 1)  # scale
            output = self.round(output / scale) * scale  # 量化/反量化
            output = max_w * (2 * output - 1)
        return output

def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred - tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred - tgt).abs().pow(p).mean()