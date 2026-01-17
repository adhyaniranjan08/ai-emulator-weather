import torch
import numpy as np

from live_inference.live_fetch import fetch_last_6_hours
from live_inference.normalize import normalize_features

from models.mlp import MLPEmulator
from models.cnn import CNNEmulator
from models.lstm import LSTMEmulator
from models.gru import GRUEmulator
from models.cnn_lstm import CNNLSTMEmulator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "MLP": {
        "builder": lambda: MLPEmulator(
            input_size=30,        # 6 timesteps × 5 features
            num_reg_outputs=3,
            num_cls_outputs=5
        ),
        "ckpt": "mlp_emulator.pt"
    },
    "CNN": {
        "ckpt": "cnn_emulator.pt",
        "builder": lambda: CNNEmulator(
            num_features=5,
            num_reg_outputs=3,
            num_cls_outputs=5
        )
    },
    "LSTM": {
        "ckpt": "lstm_emulator.pt",
        "builder": lambda: LSTMEmulator(
            num_features=5,
            hidden_size=64,
            num_reg_outputs=3,
            num_cls_outputs=5
        )
    },
    "GRU": {
        "ckpt": "gru_emulator.pt",
        "builder": lambda: GRUEmulator(
            num_features=5,
            hidden_size=64,
            num_reg_outputs=3,
            num_cls_outputs=5
        )
    },
    "CNN-LSTM": {
        "ckpt": "cnn_lstm_emulator.pt",
        "builder": lambda: CNNLSTMEmulator(
            num_features=5,
            hidden_size=64,
            num_reg_outputs=3,
            num_cls_outputs=5
        )
    },
}


def predict_with_model(model_cfg, X):
    model = model_cfg["builder"]().to(DEVICE)
    model.load_state_dict(torch.load(model_cfg["ckpt"], map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        reg_out, cls_out = model(X)

    reg_out = reg_out.cpu().numpy()[0]
    cls_probs = torch.sigmoid(cls_out).cpu().numpy()[0]

    # physical constraints
    reg_out[0] = np.maximum(0.0, reg_out[0])
    reg_out[2] = np.abs(reg_out[2])

    return reg_out, cls_probs


def compare_models(lat, lon):
    X = fetch_last_6_hours(lat, lon)
    X = normalize_features(X)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    results = {}

    for name, cfg in MODELS.items():
        reg, cls_probs = predict_with_model(cfg, X)
        results[name] = (reg, cls_probs)

    return results


if __name__ == "__main__":
    lat, lon = 12.97, 77.59  # Bangalore

    results = compare_models(lat, lon)

    for model, (reg, cls) in results.items():
        print(f"\n🔹 {model}")
        print(f"Rainfall: {reg[0]:.2f} mm")
        print(f"Temperature: {reg[1]:.2f} °C")
        print(f"Wind speed: {reg[2]:.2f} m/s")
        print(f"Coldwave prob: {cls[3]:.2f}")
