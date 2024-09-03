def conv3d(in_channels: int,
           out_channels: int,
           kernel_size: int = 3,
           stride: int = 1,
           dilation: int = 1,
           padding: int = 0,
           bias: bool = False,
           transpose: bool = False,
           indice_key: str = None,
           backend: str = 'torchsparse',
           kmap_mode: str = 'hashmap') -> None:
    if backend == 'torchsparse' or backend == 'torchsparse-1.4.0':
        import torchsparse.nn as spnn
        if backend == 'torchsparse':
            return spnn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, transpose)
        else:
            return spnn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias, transpose)
    elif backend == 'ME':
        import MinkowskiEngine as ME
        from .minkowski import MyMinkowskiConvolution
        if transpose:
            return ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size, stride,
                                                dilation, bias, dimension=3)
        else:
            return MyMinkowskiConvolution(in_channels, out_channels, kernel_size, stride, dilation,
                                        bias, dimension=3)
    elif backend == 'spconv':
        import spconv.pytorch as spconv
        # import spconv as spconv_core
        # spconv_core.constants.SPCONV_ALLOW_TF32 = True

        if transpose:
            return spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key,
                                            bias=bias)
        else:
            if stride == 1:
                return spconv.SubMConv3d(in_channels, out_channels, kernel_size, stride, dilation=dilation,
                                    bias=bias, indice_key=indice_key)
            else:
                return spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride, dilation=dilation, padding=padding,
                                    bias=bias, indice_key=indice_key)
    else:
        raise Exception(f"{backend} backend not supported")
