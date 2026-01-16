import torch
import torch.nn as nn


class GRUEmulator(nn.Module):
    def __init__(self, num_features, hidden_size, num_reg_outputs, num_cls_outputs):
        super().__init__()

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU()
        )

        self.reg_head = nn.Linear(64, num_reg_outputs)
        self.cls_head = nn.Linear(64, num_cls_outputs)

    def forward(self, x):
        # x: (batch, time, features)
        _, h_n = self.gru(x)
        features = h_n[-1]  # last hidden state

        features = self.fc(features)

        reg_out = self.reg_head(features)
        cls_out = self.cls_head(features)

        return reg_out, cls_out
