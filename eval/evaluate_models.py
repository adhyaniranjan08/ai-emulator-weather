import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support

from datasets.window_dataset import WeatherWindowDataset
from models.mlp import MLPEmulator
from models.cnn import CNNEmulator
from models.lstm import LSTMEmulator
from models.gru import GRUEmulator
from models.cnn_lstm import CNNLSTMEmulator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "data/processed/nasa_power_labeled.csv"
BATCH_SIZE = 256

MODEL_CONFIGS = {
    "MLP": {
        "model": lambda: MLPEmulator(6 * 5, 3, 5),
        "path": "mlp_emulator.pt"
    },
    "CNN": {
        "model": lambda: CNNEmulator(5, 3, 5),
        "path": "cnn_emulator.pt"
    },
    "LSTM": {
        "model": lambda: LSTMEmulator(5, 64, 3, 5),
        "path": "lstm_emulator.pt"
    },
    "GRU": {
        "model": lambda: GRUEmulator(5, 64, 3, 5),
        "path": "gru_emulator.pt"
    },
    "CNN-LSTM": {
        "model": lambda: CNNLSTMEmulator(5, 64, 3, 5),
        "path": "cnn_lstm_emulator.pt"
    }
}


def get_test_loader():
    dataset = WeatherWindowDataset(DATASET_PATH)
    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    _, _, test_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    return DataLoader(test_ds, batch_size=BATCH_SIZE)


def evaluate():
    test_loader = get_test_loader()
    results = []

    for name, cfg in MODEL_CONFIGS.items():
        print(f"\nEvaluating {name}...")

        model = cfg["model"]().to(DEVICE)
        model.load_state_dict(torch.load(cfg["path"], map_location=DEVICE))
        model.eval()

        y_reg_true, y_reg_pred = [], []
        y_cls_true, y_cls_pred = [], []

        with torch.no_grad():
            for x, y_reg, y_cls in test_loader:
                x = x.to(DEVICE)
                pred_reg, pred_cls = model(x)

                y_reg_true.append(y_reg.numpy())
                y_reg_pred.append(pred_reg.cpu().numpy())

                y_cls_true.append(y_cls.numpy())
                y_cls_pred.append(torch.sigmoid(pred_cls).cpu().numpy())

        y_reg_true = np.vstack(y_reg_true)
        y_reg_pred = np.vstack(y_reg_pred)

        y_cls_true = np.vstack(y_cls_true)
        y_cls_pred = np.vstack(y_cls_pred) > 0.5

        rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
        mae = mean_absolute_error(y_reg_true, y_reg_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_cls_true, y_cls_pred, average="macro", zero_division=0
        )

        results.append({
            "Model": name,
            "RMSE": rmse,
            "MAE": mae,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

    df = pd.DataFrame(results)
    df.to_csv("eval/model_comparison.csv", index=False)
    print("\nSaved results to eval/model_comparison.csv")
    print(df)


if __name__ == "__main__":
    evaluate()
