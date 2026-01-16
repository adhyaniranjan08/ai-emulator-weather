import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class WeatherWindowDataset(Dataset):
    def __init__(self, csv_path, window_size=6):
        self.df = pd.read_csv(csv_path, parse_dates=["time"])
        self.window_size = window_size

        self.feature_cols = ["rain", "temp", "wind", "humidity", "pressure"]
        self.reg_targets = ["rain", "temp", "wind"]
        self.cls_targets = [
            "cloudburst", "thunderstorm",
            "heatwave", "coldwave", "cyclone_like"
        ]

        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        for city in self.df["city"].unique():
            city_df = self.df[self.df["city"] == city].sort_values("time")

            X = city_df[self.feature_cols].values
            y_reg = city_df[self.reg_targets].values
            y_cls = city_df[self.cls_targets].values

            for i in range(self.window_size, len(city_df)):
                samples.append((
                    X[i - self.window_size:i],
                    y_reg[i],
                    y_cls[i]
                ))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y_reg, y_cls = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_reg, dtype=torch.float32),
            torch.tensor(y_cls, dtype=torch.float32),
        )
