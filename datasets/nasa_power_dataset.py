import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


CITY_FILES = {
    "bangalore": "bangalore_power.csv",
    "mumbai": "mumbai_power.csv",
    "chennai": "chennai_power.csv",
    "delhi": "delhi_power.csv",
}


FEATURE_COLUMNS = {
    "PRECTOTCORR": "rain",
    "T2M": "temp",
    "WS10M": "wind",
    "RH2M": "humidity",
    "PS": "pressure",
}


def load_and_clean_city(city_name, file_name):
    df = pd.read_csv(
        RAW_DIR / file_name,
        skiprows=13
    )

    # Create time column from NASA POWER fields
    df["time"] = pd.to_datetime(
    dict(
        year=df["YEAR"],
        month=df["MO"],
        day=df["DY"],
        hour=df["HR"]
    )
)

    df = df.drop(columns=["YEAR", "MO", "DY", "HR"])

    df = df.sort_values("time")

    # Keep only required features
    df = df[list(FEATURE_COLUMNS.keys()) + ["time"]]
    df = df.rename(columns=FEATURE_COLUMNS)

    df["city"] = city_name
    df = df.dropna()

    return df



def normalize_features(df, feature_cols):
    """Normalize features using Z-score"""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df


def main():
    city_dfs = []

    for city, file in CITY_FILES.items():
        print(f"Loading {city}...")
        city_df = load_and_clean_city(city, file)
        city_dfs.append(city_df)

    # Merge all cities
    full_df = pd.concat(city_dfs, ignore_index=True)

    # Save RAW merged data (for event labeling)
    raw_output = PROCESSED_DIR / "nasa_power_merged_raw.csv"
    full_df.to_csv(raw_output, index=False)
    print("Saved RAW merged dataset to:", raw_output)


    # Normalize features
    feature_cols = ["rain", "temp", "wind", "humidity", "pressure"]
    full_df = normalize_features(full_df, feature_cols)

    # Save processed dataset
    output_path = PROCESSED_DIR / "nasa_power_merged.csv"
    full_df.to_csv(output_path, index=False)

    print("Merged dataset saved to:", output_path)
    print("Final shape:", full_df.shape)
    print("Cities:", full_df["city"].unique())


if __name__ == "__main__":
    main()
