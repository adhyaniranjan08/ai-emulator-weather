import torch
import torch.nn as nn


class TransformerEmulator(nn.Module):
    def __init__(
        self,
        num_features,
        d_model,
        nhead,
        num_layers,
        num_reg_outputs,
        num_cls_outputs
    ):
        super().__init__()

        self.input_proj = nn.Linear(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU()
        )

        self.reg_head = nn.Linear(64, num_reg_outputs)
        self.cls_head = nn.Linear(64, num_cls_outputs)

    def forward(self, x):
        # x: (batch, time, features)
        x = self.input_proj(x)
        x = self.transformer(x)

        features = x[:, -1]  # last time step

        features = self.fc(features)

        reg_out = self.reg_head(features)
        cls_out = self.cls_head(features)

        return reg_out, cls_out
