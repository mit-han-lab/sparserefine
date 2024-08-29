import torch
from torch import nn

__all__ = ["EntropyEnsembler"]


class EntropyEnsembler(nn.Module):
    def forward(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        p1 = torch.softmax(l1, dim=-1)
        e1 = -torch.sum(p1 * torch.log(p1.clamp(min=1e-5)), dim=-1, keepdim=True)
        p2 = torch.softmax(l2, dim=-1)
        e2 = -torch.sum(p2 * torch.log(p2.clamp(min=1e-5)), dim=-1, keepdim=True)
        return torch.where(e1 < e2, l1, l2)
