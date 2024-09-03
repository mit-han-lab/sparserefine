from typing import Any, Dict

import torch
from torch import nn

__all__ = ["LearnableSelector"]


class LearnableSelector(nn.Module):
    def __init__(self, threshold: float, num_classes: int, percentage: float = 0.118) -> None:
        super().__init__()
        self.threshold = threshold
        self.mlp_hidden_size = 32
        self.out_channels = 16
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(num_classes+self.out_channels, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        
        image_features = self.conv(inputs["image"].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        combined_input = torch.cat((inputs["logits"],image_features), dim=-1)

        logit_mask = self.mlp(combined_input).squeeze(-1)

        return self.sigmoid(logit_mask) > self.threshold
