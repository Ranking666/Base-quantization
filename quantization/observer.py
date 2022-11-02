
import torch
import torch.nn as nn

class ObserverBase(nn.Module):
    def __init__(self):
        super(ObserverBase, self).__init__()
        # self.level = level

    # def update_range(self, input):
    #     raise NotImplementedError
    def update_range(self, min_val, max_val):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, input):
        if self.level == "L":  # layer级(activation/weight)
            min_val = torch.min(input)
            max_val = torch.max(input)
        elif self.level == "C":  # channel级(conv_weight)
            input = torch.flatten(input, start_dim=1)
            min_val = torch.min(input, 1)[0]
            max_val = torch.max(input, 1)[0]
        elif self.level == "FC":  # channel级(fc_weight)
            min_val = torch.min(input, 1, keepdim=True)[0]
            max_val = torch.max(input, 1, keepdim=True)[0]
        # self.update_range(input)
        self.update_range(min_val, max_val)

        return input

# class MinMaxObserver_PerTensor(ObserverBase):
#     def __init__(self, out_channels):
#         super(MinMaxObserver_PerTensor, self).__init__()
#         self.out_channels = out_channels
#         self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
#         self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

#     def update_range(self, input):
#         min_val = torch.min(input)
#         max_val = torch.max(input)

#         self.min_val.copy_(min_val)
#         self.max_val.copy_(max_val)

# class EMAMinMaxObserver_PerTensor(ObserverBase):
#     def __init__(self, out_channels, momentum=0.1):
#         super(EMAMinMaxObserver_PerTensor, self).__init__()
#         self.momentum = momentum
#         self.out_channels = out_channels
#         self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
#         self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

#     def update_range(self, input):
#         min_val_cur = torch.min(input)
#         max_val_cur = torch.max(input)

#         min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
#         max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur

#         self.min_val.copy_(min_val)
#         self.max_val.copy_(max_val)

class MinMaxObserver(ObserverBase):
    def __init__(self, out_channels, level):
        super(MinMaxObserver, self).__init__()
        self.out_channels = out_channels
        self.level = level
        self.num_flag = 0

        if self.level == 'L':
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

class EMAMinMaxObserver(ObserverBase):
    def __init__(self, out_channels, level, momentum=0.1):
        super(EMAMinMaxObserver, self).__init__()
        self.momentum = momentum
        self.level = level
        self.num_flag = 0
        self.out_channels = out_channels
        if self.level == 'L':
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        # min_val_cur = torch.min(input)
        # max_val_cur = torch.max(input)

        # min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
        # max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur

        # self.min_val.copy_(min_val)
        # self.max_val.copy_(max_val)
        if self.level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)


class OmseObserver(ObserverBase):

    def __init__(self, out_channels, level):
        super(OmseObserver, self).__init__()

        self.out_channels = out_channels
        self.level = level
        self.num_flag = 0

        if self.level == 'L':
            self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
            self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        elif self.level == "C":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
            )
        elif self.level == "FC":
            self.register_buffer(
                "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )
            self.register_buffer(
                "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            )

    def update_range(self, min_val_cur, max_val_cur):
        if self.level == "C":
            min_val_cur.resize_(self.min_val.shape)
            max_val_cur.resize_(self.max_val.shape)
        if self.num_flag == 0:
            self.num_flag += 1
            min_val = min_val_cur
            max_val = max_val_cur
        else:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

    # def get_quantization_params(self, inputs):
    #     max_val = self.max_val
    #     min_val = self.min_val
    #     qmax = self.bit_type.upper_bound
    #     qmin = self.bit_type.lower_bound

    #     best_score = 1e+10
    #     for i in range(90):
    #         new_max = max_val * (1.0 - (i * 0.01))
    #         new_min = min_val * (1.0 - (i * 0.01))
    #         new_scale = (new_max - new_min) / float(qmax - qmin)
    #         new_scale.clamp_(self.eps)
    #         new_zero_point = qmin - torch.round(new_min / new_scale)
    #         new_zero_point.clamp_(qmin, qmax)
    #         inputs_q = ((inputs / new_scale + new_zero_point).round().clamp(
    #             qmin, qmax) - new_zero_point) * new_scale
    #         # L_p norm minimization as described in LAPQ
    #         # https://arxiv.org/abs/1911.07190
    #         score = lp_loss(inputs, inputs_q, p=2.0, reduction='all')
    #         if score < best_score:
    #             best_score = score
    #             self.max_val = new_max
    #             self.min_val = new_min
    #             scale = new_scale
    #             zero_point = new_zero_point
    #     return scale, zero_point


class HistogramObserver(nn.Module):
    def __init__(self,out_channels, level, momentum=0.1, percentile=0.9999):
        super(HistogramObserver, self).__init__()
        self.momentum = momentum
        self.percentile = percentile
        self.num_flag = 0
        self.level = level
        self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))

        # if self.level == 'L':
        #     self.register_buffer("min_val", torch.zeros((1), dtype=torch.float32))
        #     self.register_buffer("max_val", torch.zeros((1), dtype=torch.float32))
        # elif self.level == "C":
        #     self.register_buffer(
        #         "min_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
        #     )
        #     self.register_buffer(
        #         "max_val", torch.zeros((out_channels, 1, 1, 1), dtype=torch.float32)
        #     )
        # elif self.level == "FC":
        #     self.register_buffer(
        #         "min_val", torch.zeros((out_channels, 1), dtype=torch.float32)
        #     )
        #     self.register_buffer(
        #         "max_val", torch.zeros((out_channels, 1), dtype=torch.float32)
            # )

    @torch.no_grad()
    def forward(self, input):
        # MovingAveragePercentileCalibrator
        # PercentileCalibrator
        max_val_cur = torch.kthvalue(
            input.abs().view(-1), int(self.percentile * input.view(-1).size(0)), dim=0
        )[0]
        min_val_cur = torch.min(input)
        # MovingAverage
        if self.num_flag == 0:
            self.num_flag += 1
            max_val = max_val_cur
            min_val = min_val_cur         
        else:
            max_val = (1 - self.momentum) * self.max_val + self.momentum * max_val_cur
            min_val = (1 - self.momentum) * self.min_val + self.momentum * min_val_cur
        self.max_val.copy_(max_val)
        self.min_val.copy_(min_val)
