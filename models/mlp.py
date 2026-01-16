import torch
import torch.nn as nn


class MLPEmulator(nn.Module):
    def __init__(self, input_size, num_reg_outputs, num_cls_outputs):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.reg_head = nn.Linear(64, num_reg_outputs)
        self.cls_head = nn.Linear(64, num_cls_outputs)

    def forward(self, x):
        # x: (batch, time, features)
        x = x.view(x.size(0), -1)  # flatten
        features = self.shared(x)

        reg_out = self.reg_head(features)
        cls_out = self.cls_head(features)

        return reg_out, cls_out
