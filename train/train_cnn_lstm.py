import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.window_dataset import WeatherWindowDataset
from models.cnn_lstm import CNNLSTMEmulator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "data/processed/nasa_power_labeled.csv"

EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
HIDDEN_SIZE = 64


def main():
    dataset = WeatherWindowDataset(DATASET_PATH)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, _ = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = CNNLSTMEmulator(
        num_features=5,
        hidden_size=HIDDEN_SIZE,
        num_reg_outputs=3,
        num_cls_outputs=5
    ).to(DEVICE)

    reg_loss_fn = nn.MSELoss()
    cls_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for x, y_reg, y_cls in train_loader:
            x, y_reg, y_cls = x.to(DEVICE), y_reg.to(DEVICE), y_cls.to(DEVICE)

            optimizer.zero_grad()
            pred_reg, pred_cls = model(x)

            loss = reg_loss_fn(pred_reg, y_reg) + cls_loss_fn(pred_cls, y_cls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y_reg, y_cls in val_loader:
                x, y_reg, y_cls = x.to(DEVICE), y_reg.to(DEVICE), y_cls.to(DEVICE)
                pred_reg, pred_cls = model(x)
                val_loss += (
                    reg_loss_fn(pred_reg, y_reg) +
                    cls_loss_fn(pred_cls, y_cls)
                ).item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

    torch.save(model.state_dict(), "cnn_lstm_emulator.pt")
    print("CNN-LSTM model saved.")


if __name__ == "__main__":
    main()
