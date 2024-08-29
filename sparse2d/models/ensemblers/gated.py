import torch
from torch import nn

__all__ = ["GatedEnsembler"]


class GatedEnsembler(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear((num_classes + 1) * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.fuser = nn.Sequential(
            nn.Linear(num_classes, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        p1 = torch.softmax(l1, dim=-1)
        e1 = -torch.sum(p1 * torch.log(p1.clamp(min=1e-5)), dim=-1, keepdim=True)
        x1 = torch.cat([p1, e1], dim=-1)

        p2 = torch.softmax(l2, dim=-1)
        e2 = -torch.sum(p2 * torch.log(p2.clamp(min=1e-5)), dim=-1, keepdim=True)
        x2 = torch.cat([p2, e2], dim=-1)

        x = torch.cat([x1, x2], dim=-1)
        w = torch.softmax(self.attn(x), dim=-1)
        y = torch.sum(torch.stack([l1, l2], dim=-1) * w.unsqueeze(1), dim=-1)
        return self.fuser(y)
