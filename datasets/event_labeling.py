import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DIR / "nasa_power_merged_raw.csv"

OUTPUT_FILE = PROCESSED_DIR / "nasa_power_labeled.csv"

def add_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure sorted by time per city
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # 1) Cloudburst
    df["cloudburst"] = (df["rain"] >= 100).astype(int)

    # 2) Thunderstorm (proxy-based)
    df["thunderstorm"] = (
        (df["humidity"] >= 80) &
        (df["wind"] >= 8)
    ).astype(int)

    # 3) Heat wave
    df["heatwave"] = (df["temp"] >= 40).astype(int)

    # 4) Cold wave
    df["coldwave"] = (df["temp"] <= 5).astype(int)

    # 5) Cyclone-like extreme wind event (city-specific pressure threshold)
    df["cyclone_like"] = 0
    for city in df["city"].unique():
        city_mask = df["city"] == city
        pressure_thresh = df.loc[city_mask, "pressure"].quantile(0.10)
        df.loc[
            city_mask & (df["wind"] >= 17) & (df["pressure"] <= pressure_thresh),
            "cyclone_like"
        ] = 1

    return df

def main():
    print("Loading merged dataset...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["time"])

    df_labeled = add_event_labels(df)

    df_labeled.to_csv(OUTPUT_FILE, index=False)

    print("Saved labeled dataset to:", OUTPUT_FILE)
    print("Final shape:", df_labeled.shape)
    print("\nEvent counts:")
    for col in ["cloudburst", "thunderstorm", "heatwave", "coldwave", "cyclone_like"]:
        print(f"{col}: {df_labeled[col].sum()}")

if __name__ == "__main__":
    main()
