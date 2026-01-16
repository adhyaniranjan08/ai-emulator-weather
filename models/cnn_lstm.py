import torch
import torch.nn as nn


class CNNLSTMEmulator(nn.Module):
    def __init__(self, num_features, hidden_size, num_reg_outputs, num_cls_outputs):
        super().__init__()

        # CNN over time
        self.cnn = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # LSTM over CNN features
        self.lstm = nn.LSTM(
            input_size=64,
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
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (batch, time, channels)

        _, (h_n, _) = self.lstm(x)
        features = h_n[-1]

        features = self.fc(features)

        reg_out = self.reg_head(features)
        cls_out = self.cls_head(features)

        return reg_out, cls_out
