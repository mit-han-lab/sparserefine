import torch


def cat(*args, backend: str = 'torchsparse') -> None:
    if backend == 'torchsparse' or backend == 'torchsparse-1.4.0':
        import torchsparse
        return torchsparse.cat(*args)
    elif backend == 'ME':
        import MinkowskiEngine as ME
        if len(args) == 1:
            args = args[0]
        return ME.cat(*args)
    elif backend == 'spconv':
        import spconv.pytorch as spconv
        def spconv_cat(input_list):
            assert len(input_list) > 0
            inputs = input_list[0]
            output_tensor = spconv.SparseConvTensor(
                    torch.cat([i.features for i in input_list], 1), inputs.indices,
                    inputs.spatial_shape, inputs.batch_size)
            output_tensor.indice_dict = inputs.indice_dict
            output_tensor.grid = inputs.grid
            return output_tensor
        return spconv_cat(*args)
    else:
        raise Exception(f"{backend} backend not supported")