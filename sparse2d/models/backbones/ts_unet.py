from torch import nn
import torchsparse
from torchsparse import nn as spnn
from torchsparse.nn.functional import relu
import sys
from collections import OrderedDict
from . import ts_tables
from .wrapper import wrapper as wrapper  

__all__ = ["TorchSparseUNet"]


class SparseSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(SparseSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)
        self.wrapper = wrapper.Wrapper(backend='torchsparse')

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)


def res_block(m, a, b):
    m.add(
        ts_tables.ConcatTable()
        .add(ts_tables.Identity() if a == b else m.wrapper.conv3d(a, b, kernel_size = (1, 1, 1), bias=False))
        .add(
            SparseSequential(
                m.wrapper.bn(a),
                m.wrapper.relu(),
                m.wrapper.conv3d(a, b, kernel_size = (3, 3, 1), padding = (1, 1, 0), bias=False),
                m.wrapper.bn(b),
                m.wrapper.relu(),
                m.wrapper.conv3d(b, b, kernel_size = (3, 3, 1), padding = (1, 1, 0), bias=False),
            )
        )
    )
    m.add(ts_tables.AddTable())


def U(reps, nPlanes, in_channels: int = -1):
    m = SparseSequential()
    for _ in range(reps[0]):
        res_block(m, in_channels if in_channels != -1 else nPlanes[0], nPlanes[0])
        in_channels = -1

    if len(nPlanes) > 1:
        m.add(
            ts_tables.ConcatTable()
            .add(ts_tables.Identity())
            .add(
                SparseSequential(
                    m.wrapper.bn(nPlanes[0]),
                    m.wrapper.relu(),
                    m.wrapper.conv3d(nPlanes[0], nPlanes[1], kernel_size = (3, 3, 1), stride=(2, 2, 1), bias=False),
                    U(reps[1:], nPlanes[1:]),
                    m.wrapper.bn(nPlanes[1]),
                    m.wrapper.relu(),
                    m.wrapper.conv3d(nPlanes[1], nPlanes[0], kernel_size = (3, 3, 1), stride=(2, 2, 1), bias=False, transpose = True),
                )
            )
        )

        m.add(ts_tables.JoinTable())
        for i in range(reps[0]):
           res_block(m, nPlanes[0] * (2 if i == 0 else 1), nPlanes[0])
    
    return m


class TorchSparseUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reps,
        nPlanes,
    ) -> None:
        super().__init__()
        self.wrapper = wrapper.Wrapper(backend='torchsparse')
        self.net = nn.Sequential(
            self.wrapper.conv3d(in_channels = in_channels, 
                                out_channels = nPlanes[0],
                                kernel_size = (3, 3, 1),
                                padding = (1, 1, 0),
                                bias=False,
                                ),
            U(reps, nPlanes),
            self.wrapper.bn(nPlanes[0]),
            self.wrapper.relu(),
        )

    def forward(self, x: torchsparse.SparseTensor) -> torchsparse.SparseTensor:
        return self.net(x)