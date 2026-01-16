import torch
import torch.nn as nn


class CNNEmulator(nn.Module):
    def __init__(self, num_features, num_reg_outputs, num_cls_outputs):
        super().__init__()

        # Input shape: (batch, time, features)
        # We transpose to: (batch, features, time)
        self.conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # collapse time dimension
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.reg_head = nn.Linear(64, num_reg_outputs)
        self.cls_head = nn.Linear(64, num_cls_outputs)

    def forward(self, x):
        # x: (batch, time, features)
        x = x.transpose(1, 2)  # (batch, features, time)
        features = self.conv(x).squeeze(-1)

        features = self.fc(features)

        reg_out = self.reg_head(features)
        cls_out = self.cls_head(features)

        return reg_out, cls_out
