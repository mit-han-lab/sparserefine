from typing import List
import torch
from torch import nn
import torchsparse

class ConcatTable(nn.Module):
    def forward(self, input):
        return [module(input) for module in self._modules.values()]

    def add(self, module):
        self._modules[str(len(self._modules))] = module
        return self


# TODO: Reimplement with TorchSparse functional.
class AddTable(nn.Module): 
    def forward(self, input: List[torchsparse.SparseTensor]):
        msg = "Can't use AddTable in two SparseTensor with different coords."
        for ten in input:
            assert ten.F.shape[1] == input[0].F.shape[1], msg
            assert ten.C.shape[0] == input[0].C.shape[0], msg
        output = torchsparse.SparseTensor(sum([i.F for i in input]), 
                                          input[0].C, 
                                          stride = input[0].stride, 
                                          spatial_range = input[0].spatial_range)
        output._caches = input[0]._caches
        output._caches.cmaps.setdefault(output.stride, (output.coords, output.spatial_range))
        return output



# TODO: Reimplement with TorchSparse functional.
class JoinTable(nn.Module):
    def forward(self, input: List[torchsparse.SparseTensor]):
        msg = "Can't use JoinTable in two SparseTensor with different coords."
        for ten in input:
            assert ten.F.shape[1] == input[0].F.shape[1], msg
            assert ten.C.shape[0] == input[0].C.shape[0], msg

        output = torchsparse.SparseTensor(torch.cat([i.F for i in input], 1),
                                          input[0].C, 
                                          stride = input[0].stride, 
                                          spatial_range = input[0].spatial_range)

        output._caches = input[0]._caches
        output._caches.cmaps.setdefault(output.stride, (output.coords, output.spatial_range))
        return output


class Identity(nn.Module):
    def forward(self, input):
        return input