import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

from datasets.window_dataset import WeatherWindowDataset
from models.mlp import MLPEmulator
from models.cnn_lstm import CNNLSTMEmulator
from models.gru import GRUEmulator
from models.lstm import LSTMEmulator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "data/processed/nasa_power_labeled.csv"
BATCH_SIZE = 256

MODELS = {
    "MLP": (MLPEmulator(30, 3, 5), "mlp_emulator.pt"),
    "CNN-LSTM": (CNNLSTMEmulator(5, 64, 3, 5), "cnn_lstm_emulator.pt"),
    "LSTM": (LSTMEmulator(5, 64, 3, 5), "lstm_emulator.pt"),
    "GRU": (GRUEmulator(5, 64, 3, 5), "gru_emulator.pt"),
}

def evaluate():
    dataset = WeatherWindowDataset(DATASET_PATH)
    test_size = int(0.15 * len(dataset))
    _, _, test_ds = torch.utils.data.random_split(
        dataset,
        [len(dataset) - 2 * test_size, test_size, test_size]
    )

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    results = {}

    for name, (model, ckpt) in MODELS.items():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        y_cls_true, y_cls_pred = [], []
        y_reg_true, y_reg_pred = [], []

        with torch.no_grad():
            for x, y_reg, y_cls in loader:
                x, y_reg, y_cls = x.to(DEVICE), y_reg.to(DEVICE), y_cls.to(DEVICE)
                reg_out, cls_out = model(x)

                y_cls_true.append(y_cls.cpu().numpy())
                y_cls_pred.append((torch.sigmoid(cls_out) > 0.5).cpu().numpy())

                y_reg_true.append(y_reg.cpu().numpy())
                y_reg_pred.append(reg_out.cpu().numpy())

        y_cls_true = np.vstack(y_cls_true)
        y_cls_pred = np.vstack(y_cls_pred)
        y_reg_true = np.vstack(y_reg_true)
        y_reg_pred = np.vstack(y_reg_pred)

        results[name] = {
            "Accuracy": accuracy_score(y_cls_true.flatten(), y_cls_pred.flatten()),
            "Precision": precision_score(y_cls_true.flatten(), y_cls_pred.flatten(), zero_division=0),
            "Recall": recall_score(y_cls_true.flatten(), y_cls_pred.flatten(), zero_division=0),
            "F1": f1_score(y_cls_true.flatten(), y_cls_pred.flatten(), zero_division=0),
            "MAE": mean_absolute_error(y_reg_true, y_reg_pred)
        }

    return results


if __name__ == "__main__":
    metrics = evaluate()
    for model, vals in metrics.items():
        print(f"\nModel: {model}")
        for k, v in vals.items():
            print(f"{k}: {v:.4f}")
