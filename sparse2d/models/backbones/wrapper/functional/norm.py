import torch.nn as nn


def batchnorm(num_features: int,
              *,
              eps: float = 1e-5,
              momentum: float = 0.1, 
              backend: str = 'torchsparse') -> None:
    if backend == 'torchsparse' or backend == 'torchsparse-1.4.0':
        import torchsparse.nn as spnn
        return spnn.BatchNorm(num_features=num_features, eps=eps, momentum=momentum)
    elif backend == 'ME':
        import MinkowskiEngine as ME
        return ME.MinkowskiBatchNorm(num_features=num_features, eps=eps, momentum=momentum)
    elif backend == 'spconv':
        return nn.BatchNorm1d(num_features=num_features, eps=eps, momentum=momentum)
    else:
        raise Exception(f"{backend} backend not supported")
