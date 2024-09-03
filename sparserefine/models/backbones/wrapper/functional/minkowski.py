import torch
import MinkowskiEngine as ME
#import torchsparse.nn.functional as F
from torchsparse.nn.utils import get_kernel_offsets
from torchsparse.utils import make_ntuple
from MinkowskiSparseTensor import SparseTensor, _get_coordinate_map_key
from typing import Union, Tuple


def spdownsample(
        coords: torch.Tensor,
        stride: Union[int, Tuple[int, ...]] = 2,
        kernel_size: Union[int, Tuple[int, ...]] = 2,
        tensor_stride: Union[int, Tuple[int, ...]] = 1) -> torch.Tensor:
    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    tensor_stride = make_ntuple(tensor_stride, ndim=3)

    sample_stride = [stride[k] * tensor_stride[k] for k in range(3)]
    sample_stride = torch.tensor(sample_stride,
                                 dtype=torch.int,
                                 device=coords.device).unsqueeze(dim=0)

    if all(stride[k] in [1, kernel_size[k]] for k in range(3)):
        coords = coords.clone()
        coords[:, :3] = coords[:, :3] // sample_stride * sample_stride
    else:
        offsets = get_kernel_offsets(kernel_size,
                                     tensor_stride,
                                     device=coords.device)
        kernel_volume = offsets.size(0)

        coords_min = torch.min(coords[:, :3], dim=0, keepdim=True).values

        x = coords[:, :3].unsqueeze(dim=1).repeat(1, kernel_volume, 1) + offsets
        b = coords[:, 3:].repeat(1, kernel_volume)
        coords = torch.cat([x.view(-1, 3), b.view(-1, 1)], dim=1)

        # TODO(Zhijian): We need to also filter `coords` based on `coords_max`.
        mask = (coords[:, :3] % sample_stride == 0)
        mask &= (coords[:, :3] >= coords_min)
        mask = torch.all(mask, dim=1)
        coords = coords[mask]

    # This makes sure that the points will be ordered with respect to the batch
    # index, but this will not affect the correctness of the result.
    coords = coords[:, [3, 0, 1, 2]]
    coords = torch.unique(coords, dim=0)
    coords = coords[:, [1, 2, 3, 0]]
    return coords



class MyMinkowskiConvolution(ME.MinkowskiConvolution):
    def __init__(self, 
        in_channels,
        out_channels,
        kernel_size=-1,
        stride=1,
        dilation=1,
        bias=False,
        kernel_generator=None,
        dimension=3):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            kernel_generator=kernel_generator,
            dimension=dimension
        )
        self.kernel_size = make_ntuple(kernel_size, dimension)
        self.stride = make_ntuple(stride, dimension)

    def forward(self, input):
        if self.stride != (1, 1, 1) and self.stride != self.kernel_size:
            cm = input._manager
            new_c = spdownsample(input.C[:, [1, 2, 3, 0]], self.stride, self.kernel_size, input.tensor_stride)
            new_tensor_stride = [self.stride[i] * input.tensor_stride[i] for i in range(len(self.stride))]
            #out_coords_key = cm.insert_and_map(new_c, tensor_stride=new_tensor_stride)
            #out = super().forward(input, out_coords_key)
            out_coordinate_map_key = _get_coordinate_map_key(
                input, new_c[:, [3, 0, 1, 2]], new_tensor_stride
            )
            out = self.conv.apply(
                input.F,
                self.kernel,
                self.kernel_generator,
                self.convolution_mode,
                input.coordinate_map_key,
                out_coordinate_map_key,
                input._manager,
            )
            return SparseTensor(
                out,
                coordinate_map_key=out_coordinate_map_key,
                coordinate_manager=input._manager,
            )
        else:
            return super().forward(input)
