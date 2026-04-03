import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based import surrogate, neuron

from torch.nn.common_types import _size_2_t


class IF(neuron.IFNode):
    def __init__(self):
        super().__init__(v_threshold=1., v_reset=0., surrogate_function=surrogate.ATan(),
                         detach_reset=True, step_mode='m', backend='cupy', store_v_seq=False)


class LIF(neuron.LIFNode):
    def __init__(self, v_threshold, decay_input=True, v_reset=0.):
        super().__init__(tau=2., decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)

class LIFSigmoid(neuron.LIFNode):
    def __init__(self, v_threshold, decay_input=True, v_reset=0.):
        super().__init__(tau=2., decay_input=decay_input, v_threshold=v_threshold, v_reset=v_reset,
                         surrogate_function=surrogate.Sigmoid(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class PLIF(neuron.ParametricLIFNode):
    def __init__(self):
        super().__init__(init_tau=2., decay_input=True, v_threshold=1., v_reset=0.,
                         surrogate_function=surrogate.ATan(), detach_reset=True, step_mode='m',
                         backend='cupy', store_v_seq=False)


class BN(nn.Module):
    def __init__(self, num_features, v_th, alpha):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True,
                                 track_running_stats=True)
        torch.nn.init.constant_(self.bn.weight, alpha * v_th)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
        return functional.seq_to_ann_forward(x, self.bn)


class BN1d(nn.Module):
    def __init__(self, num_features, v_th, alpha):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True,
                                 track_running_stats=True)
        torch.nn.init.constant_(self.bn.weight, alpha * v_th)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not (x.dim() == 3 or x.dim() == 4):
            raise ValueError(
                f'expected x with shape [T, B, C] or [T,B, C, N], but got x with shape {x.shape}!')
        return functional.seq_to_ann_forward(x, self.bn)


class SpikingMatmul(nn.Module):
    def __init__(self, spike: str) -> None:
        super().__init__()
        assert spike == 'l' or spike == 'r' or spike == 'both'
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor):
        return torch.matmul(left, right)


class Conv3x3(layer.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
        padding_mode='reflect',
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation,
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode,
                         step_mode='m')


class Conv1x1(layer.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        bias: bool = False,
        padding_mode='reflect',
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                         dilation=1, groups=1, bias=bias, padding_mode=padding_mode, step_mode='m')

class Conv1dT(layer.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                         dilation=1, groups=1, bias=bias, padding_mode='reflect', step_mode='m')


class Linear(layer.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, step_mode='m')


class Quant(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, i, min_value, max_value):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class MultiSpike(nn.Module):
    def __init__(
            self,
            min_value=0,
            max_value=4,
            Norm=None,
    ):
        super().__init__()
        if Norm == None:
            self.Norm = max_value
        else:
            self.Norm = Norm
        self.min_value = min_value
        self.max_value = max_value

    @staticmethod
    def spike_function(x, min_value, max_value):
        return Quant.apply(x, min_value, max_value)

    def __repr__(self):
        return f"MultiSpike(Max_Value={self.max_value}, Min_Value={self.min_value}, Norm={self.Norm})"

    def forward(self, x):  # B C H W
        return self.spike_function(x, min_value=self.min_value, max_value=self.max_value)