# This file was generated by Nuitka

# Stubs included by default
from typing_extensions import Self
import torch.nn
import torch
import math
import torch.nn.functional

def squash(x: typing.Any) -> typing.Any:
    ...
class CapsLayer:
    def __init__(self: Self, input_caps: typing.Any, input_dim: typing.Any, output_caps: typing.Any, output_dim: typing.Any, device: typing.Any) -> None: ...
    def reset_parameters(self: Self) -> typing.Any: ...
    def forward(self: Self, u: typing.Any) -> typing.Any: ...
class AgreementRouting:
    def __init__(self: Self, input_caps: typing.Any, output_caps: typing.Any, device: typing.Any, n_iterations: typing.Any) -> None: ...
    def forward(self: Self, u_predict: typing.Any) -> typing.Any: ...
class DPCNNlayer:
    def __init__(self: Self) -> None: ...
    def forward(self: Self, x: typing.Any) -> typing.Any: ...
class BagAttention_Para:
    def __init__(self: Self, hidden_size: typing.Any) -> None: ...
    def forward(self: Self, encoder_outputs: typing.Any) -> typing.Any: ...
class MHCpre_model_MIL_Capsule:
    def __init__(self: Self, esm_model: typing.Any, device: typing.Any) -> None: ...
    def conv_and_norm(self: Self, x: typing.Any, conv: typing.Any, norm: typing.Any) -> typing.Any: ...
    def forward(self: Self, input_data: typing.Any, input_ids: typing.Any, device: typing.Any) -> typing.Any: ...

__name__ = ...



# Modules used internally, to allow implicit dependencies to be seen:
import torch
import math
import torch.nn
import torch.nn.functional
