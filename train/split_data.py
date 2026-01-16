from torch.utils.data import random_split
from datasets.window_dataset import WeatherWindowDataset

DATASET_PATH = "data/processed/nasa_power_labeled.csv"

def get_datasets():
    dataset = WeatherWindowDataset(DATASET_PATH)

    total = len(dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    return random_split(dataset, [train_size, val_size, test_size])


if __name__ == "__main__":
    train_ds, val_ds, test_ds = get_datasets()
    print("Train:", len(train_ds))
    print("Val:", len(val_ds))
    print("Test:", len(test_ds))
