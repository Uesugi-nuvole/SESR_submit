import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from .lsq import *

class QuanConv(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 nbit_w=8, nbit_a=8, stride=1, padding=0, norm=False, act=False,
                 dilation=1, groups=1, bias=True, quan_input=True, input_per_channel=False): # vsq_mode: normal, pw, 3d
        super(QuanConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.norm = norm
        self.act = act
        self.w = torch.zeros(self.weight.shape)
        self.b = torch.zeros(self.weight.shape[0])
        self.lsq_w  = LsqQuantizer4weight(
                        bit=self.nbit_w,
                        all_positive=False,
                        per_channel=True,
                        per_channel_num=self.weight.shape[0])
        self.quan_w = self.lsq_w
        self.quan_input = quan_input
        
        if self.quan_input:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a,
                            all_positive=True,
                            per_channel=False,
                            per_channel_num=self.weight.shape[1],
                            weight_next=True)
            self.quan_a = self.lsq_a
            
        if self.norm:
            self.bn_weight = torch.nn.parameter.Parameter(torch.ones(out_channels))
            self.bn_bias = torch.nn.parameter.Parameter(torch.zeros(out_channels))
            self.register_buffer('bn_running_mean', torch.zeros(out_channels))
            self.register_buffer('bn_running_var', torch.ones(out_channels))
            self.bn_running_mean = torch.zeros(out_channels)
            self.bn_running_var = torch.ones(out_channels)
            self.momentum = 0.1

    # @weak_script_method
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x = self.quan_a(input)
        # quantize input
        x = input
        if self.training and self.norm: 
            output = F.conv2d(x, self.weight, self.bias,
                            self.stride, self.padding, self.dilation, self.groups)
            mean_bn = output.mean([0, 2,3], keepdim=False).squeeze(0)
            var_bn = output.var([0, 2,3], keepdim=False).squeeze(0)
            output1 = F.batch_norm(output, self.bn_running_mean, self.bn_running_var, weight=self.bn_weight,
                                   bias=self.bn_bias, training=self.training, momentum=0.1, eps=1e-05)
        if not self.training and self.norm:
            mean_bn = self.bn_running_mean
            var_bn = self.bn_running_var
            
        if self.norm:
            tmp = self.bn_weight / torch.sqrt(var_bn + 1e-5)
            w = tmp.view(tmp.size()[0], 1, 1, 1) * self.weight
            b = self.bias
            if self.bias:
                b = tmp*(self.bias - mean_bn) + self.bn_bias
            else:
                b = tmp*(0 - mean_bn) + self.bn_bias
        else:
            w = self.weight
            b = self.bias
        
        weight_integer, weight_scaling_factor = self.quan_w(w)
        weight_integer = weight_integer.to(x.device)
        weight_scaling_factor = weight_scaling_factor.to(x.device)
        
        if b is not None:
            b = b.to(x.device)
            bias_integer = SymmetricQuantFunction.apply(b, 16, scale_x * weight_scaling_factor.squeeze()) * scale_x * weight_scaling_factor.squeeze()
            bias_integer = bias_integer.to(x.device)
        else:
            bias_integer = None

        output2 = F.conv2d(x, weight_integer, bias_integer, self.stride, self.padding, self.dilation, self.groups) 

        if self.training and self.norm:
            output1 = output1 - output2.detach() + output2
            
        return output2
class QuanTransConv(nn.ConvTranspose2d):

    def __init__(self, in_channels, out_channels, kernel_size,
                 nbit_w=8, nbit_a=8, stride=1, padding=0, output_padding=0,norm=False, act=False,
                 dilation=1, groups=1, bias=True, quan_input=True, input_per_channel=False): # vsq_mode: normal, pw, 3d
        super(QuanTransConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, output_padding,groups, dilation,
             bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.norm = norm
        self.act = act
        self.w = torch.zeros(self.weight.shape)
        self.b = torch.zeros(self.weight.shape[0])
        self.lsq_w  = LsqQuantizer4weight(
                        bit=self.nbit_w,
                        all_positive=False,
                        per_channel=True,
                        per_channel_num=self.weight.shape[0])
        self.quan_w = self.lsq_w
        self.quan_input = quan_input
        
        if self.quan_input:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                            bit=self.nbit_a,
                            all_positive=True,
                            per_channel=False,
                            per_channel_num=self.weight.shape[1],
                            weight_next=True)
            self.quan_a = self.lsq_a
            
        if self.norm:
            self.bn_weight = torch.nn.parameter.Parameter(torch.ones(out_channels))
            self.bn_bias = torch.nn.parameter.Parameter(torch.zeros(out_channels))
            self.register_buffer('bn_running_mean', torch.zeros(out_channels))
            self.register_buffer('bn_running_var', torch.ones(out_channels))
            self.bn_running_mean = torch.zeros(out_channels)
            self.bn_running_var = torch.ones(out_channels)
            self.momentum = 0.1

    # @weak_script_method
    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x = self.quan_a(input)
        # quantize input
        x = input
        if self.training and self.norm: 
            output = F.conv_transpose2d(x, self.weight, self.bias,
                            self.stride, self.padding,self.output_padding,self.groups,self.dilation)
            mean_bn = output.mean([0, 2,3], keepdim=False).squeeze(0)
            var_bn = output.var([0, 2,3], keepdim=False).squeeze(0)
            output1 = F.batch_norm(output, self.bn_running_mean, self.bn_running_var, weight=self.bn_weight,
                                   bias=self.bn_bias, training=self.training, momentum=0.1, eps=1e-05)
        if not self.training and self.norm:
            mean_bn = self.bn_running_mean
            var_bn = self.bn_running_var
            
        if self.norm:
            tmp = self.bn_weight / torch.sqrt(var_bn + 1e-5)
            w = tmp.view(tmp.size()[0], 1, 1, 1) * self.weight
            b = self.bias
            if self.bias:
                b = tmp*(self.bias - mean_bn) + self.bn_bias
            else:
                b = tmp*(0 - mean_bn) + self.bn_bias
        else:
            w = self.weight
            b = self.bias
        
        weight_integer, weight_scaling_factor = self.quan_w(w)
        weight_integer = weight_integer.to(x.device)
        weight_scaling_factor = weight_scaling_factor.to(x.device)
        
        if b is not None:
            b = b.to(x.device)
            bias_integer = SymmetricQuantFunction.apply(b, 16, scale_x * weight_scaling_factor.squeeze()) * scale_x * weight_scaling_factor.squeeze()
            bias_integer = bias_integer.to(x.device)
        else:
            bias_integer = None

        output2 = F.conv_transpose2d(x, weight_integer, bias_integer, self.stride, self.padding, self.output_padding, self.groups) 

        if self.training and self.norm:
            output1 = output1 - output2.detach() + output2
            
        return output2

class QuanReLU(nn.ReLU):

    def __init__(self, quan_input=True, nbit_a=8):
        super(QuanReLU, self).__init__()
        self.quan_input = quan_input
        if self.quan_input:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
            self.quan_a = self.lsq_a

    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x, = self.quan_a(input)
        return F.relu(input)

class QuanModule(nn.Module):

    def __init__(self, quan_input=True, nbit_a=8):
        super(QuanModule, self).__init__()
        self.quan_input = quan_input
        if self.quan_input:
            self.nbit_a = nbit_a
            self.lsq_a  = LsqQuantizer4input(
                                bit=self.nbit_a,
                                all_positive=False,
                                per_channel=False)
            self.quan_a = self.lsq_a

    def forward(self, input, scale_x=None):
        if self.quan_input:
            input, scale_x, = self.quan_a(input)
        return input

def linear_quantize(input, scale, zero_point, inplace=False):
    """
    Quantize floating point input tensor to integers with the given scaling factor and zeropoint.

    Parameters:
    ----------
    input: floating point input tensor to be quantized
    scale: scaling factor for quantization
    zero_pint: shift for quantization
    """
    # reshape scale and zeropoint for convolutional weights and activations
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    # reshape scale and zeropoint for linear weights
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        scale = scale.view(-1)
        zero_point = zero_point.view(-1)
    if inplace:
        input.mul_(1. / scale).add_(zero_point).round_()
        return input
    return torch.round(1. / scale * input + zero_point)


class SymmetricQuantFunction(Function):
    """
    Class to quantize the given floating-point values using symmetric quantization with given range and bitwidth.
    """

    @staticmethod
    def forward(ctx, x, k, specified_scale=None):
        """
        x: floating point tensor to be quantized
        k: quantization bitwidth
        Note that the current implementation of SymmetricQuantFunction requires pre-calculated scaling factor.
        specified_scale: pre-calculated scaling factor for the tensor x
        """
        n = 2 ** (k - 1) - 1

        if specified_scale is not None:
            scale = specified_scale
        else:
            raise ValueError("The SymmetricQuantFunction requires a pre-calculated scaling factor")

        zero_point = torch.tensor(0.).to(x.device)

        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False)

        new_quant_x = torch.clamp(new_quant_x, -n - 1, n)

        ctx.scale = scale
        return new_quant_x

    @staticmethod
    def backward(ctx, grad_output):

        scale = ctx.scale
        if len(grad_output.shape) == 4:
            scale = scale.view(-1, 1, 1, 1)
        # reshape scale and zeropoint for linear weights
        elif len(grad_output.shape) == 2:
            scale = scale.view(-1, 1)
        else:
            scale = scale.view(-1)

        return grad_output.clone() / scale, None, None, None
