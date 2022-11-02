
from . import quantizer
from . import observer

import torch
import torch.nn as nn
import torch.nn.functional as F

class QLinear(nn.Linear):
    def __init__(self, ptq, dorefa, Histogram, level, omse, adaround, bias_correction, lsq, in_features, out_features, bias=True, bit=8,
                 sign=True, **kwargs):

        super(QLinear, self).__init__(in_features, out_features, bias)

        self.bias_correction = bias_correction
        self.ptq = ptq
        if level == 'L':
            self.fc_level = 'L'
        elif level == 'C':
            self.fc_level = 'FC'

        if dorefa:

            self.weight_quantizer = quantizer.dorefa_WeightQuantizer(2)

            self.input_quantizer = quantizer.dorefa_ActivationQuantizer(32)

        elif Histogram:

            self.weight_quantizer = quantizer.AsymmetricQuantizer(bit = bit, 
                                                                            observer = observer.HistogramObserver(out_features, "L"),
                                                                            ptq = ptq,
                                                                            sign = sign)
            self.input_quantizer = quantizer.AsymmetricQuantizer(bit = bit, 
                                                                        observer = observer.HistogramObserver(out_features, "L"),
                                                                        ptq = ptq,
                                                                        sign = sign) 
        elif omse:

            self.weight_quantizer = quantizer.AsymmetricQuantizer(bit = bit, 
                                                                            observer = observer.OmseObserver(out_features, self.fc_level),
                                                                            ptq = ptq,
                                                                            sign = sign)
            self.input_quantizer = quantizer.AsymmetricQuantizer(bit = bit, 
                                                                        observer = observer.OmseObserver(out_features, "L"),
                                                                        ptq = ptq,
                                                                        sign = sign) 

        elif adaround:

            self.weight_quantizer = quantizer.AdaRoundQuantizer(bit = bit, 
                                                                            observer = observer.MinMaxObserver(out_features, self.fc_level),
                                                                            ptq = ptq,
                                                                            sign = sign)
            self.input_quantizer = quantizer.AsymmetricQuantizer(bit = bit, 
                                                                        observer = observer.EMAMinMaxObserver(out_features, "L"),
                                                                        ptq = ptq,
                                                                        sign = sign) 
        elif lsq:
            self.weight_quantizer = quantizer.LsqQuantizer(bit = bit, level = self.fc_level,
                                                                    ptq = ptq,
                                                                    sign = sign)
            self.input_quantizer = quantizer.LsqQuantizer(bit = bit,  level = "L",
                                                            ptq = ptq,
                                                            sign = sign) 
            
            self.weight_quantizer.init_from(self.weight)


        else:
            # self.weight_quantizer = quantizer.AsymmetricQuantizer_PerTensor(bit = bit, 
            #                                                                 observer = observer.MinMaxObserver_PerTensor(None),
            #                                                                 ptq = ptq,
            #                                                                 sign = sign)
            # self.input_quantizer = quantizer.AsymmetricQuantizer_PerTensor(bit = bit, 
            #                                                             observer = observer.EMAMinMaxObserver_PerTensor(None),
            #                                                             ptq = ptq,
            #                                                             sign = sign)
            self.weight_quantizer = quantizer.AsymmetricQuantizer(bit = bit, 
                                                                            observer = observer.MinMaxObserver(out_features, self.fc_level),
                                                                            ptq = ptq,
                                                                            sign = sign)
            self.input_quantizer = quantizer.AsymmetricQuantizer(bit = bit, 
                                                                        observer = observer.EMAMinMaxObserver(out_features, "L"),
                                                                        ptq = ptq,
                                                                        sign = sign)

    def forward(self, input):
        if self.ptq:
            if self.bias_correction and self.bias != None:
                self.bias_correction_quant_forward(input)

        input = self.input_quantizer(input)
        weight_quant = self.weight_quantizer(self.weight)

        output = F.linear(input, weight_quant, self.bias)
        return output
    def bias_correction_quant_forward(self, x):
            w_sim = self.weight_quantizer(self.weight)
            x_sim = self.input_quantizer(x)
            eps = F.linear(x_sim, w_sim-self.weight.data, None)
            eps = torch.mean(eps, dim=(list(range(len(eps.shape)-1))), keepdim=False)
            self.bias -= eps
            # self.bias_correction = False
