import torch
import numpy as np

from live_inference.live_fetch import fetch_last_6_hours
from live_inference.normalize import normalize_features
from models.cnn_lstm import CNNLSTMEmulator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "cnn_lstm_emulator.pt"   # best trained model


def run_live_inference(lat, lon):
    # 1. Fetch last 6 hours of live weather data
    # Shape: (6, 5)
    # Order: [rain, temp, wind, humidity, pressure]
    X = fetch_last_6_hours(lat, lon)

    # 2. Normalize INPUT FEATURES (same stats as training)
    X = normalize_features(X)

    # 3. Convert to tensor
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 6, 5)

    # 4. Load trained model
    model = CNNLSTMEmulator(
        num_features=5,
        hidden_size=64,
        num_reg_outputs=3,
        num_cls_outputs=5
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 5. Predict
    with torch.no_grad():
        reg_out, cls_out = model(X)
    reg_out = reg_out.cpu().numpy()[0] 
    # Physical constraints
    reg_out[0] = max(0.0, reg_out[0])      # rainfall ≥ 0
    reg_out[2] = abs(reg_out[2])           # wind speed ≥ 0
    

    reg_out[1] = max(-10.0, min(50.0, reg_out[1]))



    # 6. Convert outputs
    
    cls_probs = torch.sigmoid(cls_out).cpu().numpy()[0]

    return reg_out, cls_probs


if __name__ == "__main__":
    # Example: Bangalore
    lat, lon = 12.97, 77.59

    reg, cls = run_live_inference(lat, lon)

    print("\nNext-hour predictions:")
    print(f"Rainfall (mm):    {reg[0]:.2f}")
    print(f"Temperature (°C): {reg[1]:.2f}")
    print(f"Wind speed (m/s): {reg[2]:.2f}")

    events = [
        "Cloudburst",
        "Thunderstorm",
        "Heatwave",
        "Coldwave",
        "Cyclone-like"
    ]

    print("\nEvent probabilities:")
    for e, p in zip(events, cls):
        print(f"{e}: {p:.2f}")
