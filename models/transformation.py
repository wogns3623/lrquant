
import torch
import pdb
import numpy as np

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)


def smooth_ln_fcs_temporary(ln, fcs, scales, shifts):
    ln.use_temporary_parameter = True
    if not isinstance(fcs, list):
        fcs = [fcs]

    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.temp_bias = (ln.bias - shifts) / scales
    else:
        ln.temp_bias = (-1*shifts)/ scales

    ln.temp_weight = ln.weight / scales

    for fc in fcs:
        fc.use_temporary_parameter = True
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.temp_bias = fc.bias + fc.weight@shifts # B' = B + z_s * W
        else:
            fc.temp_bias = fc.weight@shifts
        fc.temp_weight = fc.weight * scales.view(1,-1) # W' = diag(s)W

# v, o, out_ hyper parameter
def smooth_fc_fc_temporary(v_proj, o_proj, scales,shifts=None):
    # only support for v_proj and out_proh now.
    v_proj.use_temporary_parameter = True
    o_proj.use_temporary_parameter = True

    if hasattr(v_proj, 'temp_weight'):
        v_proj.temp_bias = v_proj.temp_bias - shifts
        v_proj.temp_bias = v_proj.temp_bias/scales.view(-1) 
        v_proj.temp_weight = v_proj.temp_weight/scales.view(-1,1)
    else:
        v_proj.temp_bias = v_proj.bias/scales.view(-1)
        v_proj.temp_weight = v_proj.weight/scales.view(-1,1)
    
    if hasattr(o_proj, 'bias') and o_proj.bias is not None:
        o_proj.temp_bias = o_proj.bias + o_proj.weight@shifts
    else:
        o_proj.temp_bias = o_proj.weight@shifts
    o_proj.temp_weight = o_proj.weight * scales.view(1,-1)


def smooth_q_k_temporary(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = True
    k_proj.use_temporary_parameter = True

    # scales = cal_log_scales(scales, act)
    q_proj.temp_weight = q_proj.temp_weight/scales.view(-1,1)
    q_proj.temp_bias = q_proj.temp_bias/scales.view(-1)
    k_proj.temp_weight = k_proj.temp_weight*scales.view(-1,1)
    k_proj.temp_bias = k_proj.temp_bias*scales.view(-1)

def smooth_ln_fcs_inplace(ln, fcs, scales,shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]

    # scales = cal_log_scales(scales, weight, act)

    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts)
        ln.bias.div_(scales)
    else:
        del ln.bias
        ln.register_buffer('bias',(-1*shifts)/scales)

    ln.weight.div_(scales)
    for fc in fcs:
        fc.use_temporary_parameter = False
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.bias.add_(fc.weight@shifts)
        else:
            del fc.bias
            fc.register_buffer('bias',fc.weight@shifts)
        fc.weight.mul_(scales.view(1,-1))

# 
def smooth_fc_fc_inplace(fc1, fc2, scales, shifts=None):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False

    # scales = cal_log_scales(scales, weight, act)

    fc1.bias.sub_(shifts)
    fc1.bias.div_(scales.view(-1))
    fc1.weight.div_(scales.view(-1,1))
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.bias.add_(fc2.weight@shifts)
    else:
        del fc2.bias
        fc2.register_buffer('bias',fc2.weight@shifts)
    fc2.weight.mul_(scales.view(1,-1))

def smooth_q_k_inplace(q_proj, k_proj, scales):
    q_proj.use_temporary_parameter = False
    k_proj.use_temporary_parameter = False

    # scales = cal_log_scales(scales, act)

    q_proj.weight.div_(scales.view(-1,1))
    q_proj.bias.div_(scales.view(-1))
    k_proj.weight.mul_(scales.view(-1,1))
    k_proj.bias.mul_(scales.view(-1))

# tta shift

def tta_ln_fcs_inplace(ln, fcs, tta_shifts):
    ln.use_temporary_parameter = False
    if not isinstance(fcs, list):
        fcs = [fcs]
    
    # shifts = shifts + tta_shifts
    shifts = tta_shifts

    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.sub_(shifts)

    for fc in fcs:
        fc.use_temporary_parameter = False
        if hasattr(fc, 'bias') and fc.bias is not None:
            fc.bias.add_(fc.weight@shifts)
        else:
            del fc.bias
            fc.register_buffer('bias',fc.weight@shifts)


def tta_fc_fc_inplace(fc1, fc2, tta_shifts):
    # only support for v_proj and out_proh now.
    fc1.use_temporary_parameter = False
    fc2.use_temporary_parameter = False

    shifts = tta_shifts

    fc1.bias.sub_(shifts)
    
    if hasattr(fc2, 'bias') and fc2.bias is not None:
        fc2.bias.add_(fc2.weight@shifts)
    else:
        del fc2.bias
        fc2.register_buffer('bias',fc2.weight@shifts)
