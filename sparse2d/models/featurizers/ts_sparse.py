from typing import Any, Dict, List
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchsparse

__all__ = ["TorchSparseFeaturizer"]

class TorchSparseFeaturizer(nn.Module):
    def __init__(self, features: List[str], is_half: bool = True) -> None:
        super().__init__()
        self.features = features
        self.is_half = is_half

    def forward(self, inputs: Dict[str, Any], mask: torch.Tensor) -> torchsparse.SparseTensor:
        batch_size, *spatial_shape = mask.shape
        indices = mask.nonzero()

        features = []
        for name in self.features:
            if name == "xy":
                feature = indices[:, 1:3].float() / torch.tensor(spatial_shape, device=indices.device).float()
            elif name == "rgb":
                feature = inputs["image"][mask]
            elif name.startswith("p"):
                feature = torch.softmax(inputs["logits"][mask], dim=-1)
                if name.startswith("p>"):
                    feature = (feature > float(name[2:])).float()
            else:
                raise ValueError(f"Unknown feature: '{name}'")
            features.append(feature)
        features = torch.cat(features, dim=-1)

        # Pad coordinates to 3D
        indices = torch.cat([indices, torch.zeros(indices.shape[0], 1, device=indices.device)], dim=-1)

        if self.is_half:
            sp_tensor = torchsparse.SparseTensor(features.half(), indices.int())
        else:
            sp_tensor = torchsparse.SparseTensor(features.float(), indices.int())
        return sp_tensor