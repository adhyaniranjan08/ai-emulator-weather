import pandas as pd
import numpy as np

DATA_PATH = "data/processed/nasa_power_labeled.csv"

# 🔹 INPUT FEATURES (for model input normalization)
FEATURES = ["rain", "temp", "wind", "humidity", "pressure"]

# 🔹 REGRESSION TARGETS (for output denormalization)
TARGETS = ["rain", "temp", "wind"]


def get_feature_normalizer():
    df = pd.read_csv(DATA_PATH)
    mean = df[FEATURES].mean().values
    std = df[FEATURES].std().values
    return mean, std


def get_target_normalizer():
    df = pd.read_csv(DATA_PATH)
    mean = df[TARGETS].mean().values
    std = df[TARGETS].std().values
    return mean, std


def normalize_features(X):
    mean, std = get_feature_normalizer()

    # feature order: [rain, temp, wind, humidity, pressure]
    X_norm = X.copy()

    # normalize rain, wind, humidity, pressure
    for i in [0, 2, 3, 4]:
        X_norm[:, i] = (X[:, i] - mean[i]) / (std[i] + 1e-6)

    # DO NOT normalize temperature
    return X_norm


