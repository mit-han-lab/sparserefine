import torch.nn as nn
from .functional import conv3d, batchnorm, relu, cat

class Wrapper():
    def __init__(self, backend: str = 'torchsparse') -> None:
        self.backend = backend

    def conv3d(self, in_channels: int,
               out_channels: int,
               kernel_size: int = 3,
               stride: int = 1,
               dilation: int = 1,
               padding: int = 0,
               bias: bool = False,
               transpose: bool = False,
               indice_key: str = None,
               kmap_mode: str = 'hashmap') -> None:
        return conv3d(in_channels, out_channels, kernel_size, stride, dilation,
                                padding, bias, transpose, indice_key, backend=self.backend,
                                kmap_mode=kmap_mode)

    def bn(self, num_features: int,
           *,
           eps: float = 1e-5,
           momentum: float = 0.1) -> None:
        return batchnorm(num_features=num_features, eps=eps, momentum=momentum, 
                                backend=self.backend)

    def relu(self, inplace: bool = True) -> None:
        return relu(inplace=inplace, backend=self.backend)

    def cat(self, *args) -> None:
        return cat(*args, backend=self.backend)

    def sequential(self, *args):
        if self.backend != 'spconv':
            return nn.Sequential(*args)
        else:
            import spconv.pytorch as spconv
            return spconv.SparseSequential(*args)
