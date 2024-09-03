import torch.nn as nn


def relu(inplace: bool = True, backend: str = 'torchsparse') -> None:
    if backend == 'torchsparse' or backend == 'torchsparse-1.4.0':
        import torchsparse.nn as spnn
        return spnn.ReLU(inplace)
    elif backend == 'ME':
        import MinkowskiEngine as ME
        return ME.MinkowskiReLU(inplace)
    elif backend == 'spconv':
        return nn.ReLU(inplace)
    else:
        raise Exception(f"{backend} backend not supported")