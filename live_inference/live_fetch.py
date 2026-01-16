import requests
import numpy as np

def fetch_last_6_hours(lat, lon):
    """
    Fetch last 6 hours of weather data using Open-Meteo
    Returns array shape: (6, 5)
    Features:
    [temperature, rain, windspeed, humidity, pressure]
    """

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,precipitation,windspeed_10m,"
        "relativehumidity_2m,surface_pressure"
        "&past_days=1"
        "&forecast_days=1"
    )

    r = requests.get(url)
    data = r.json()["hourly"]

    temp = data["temperature_2m"][-6:]
    rain = data["precipitation"][-6:]
    wind = data["windspeed_10m"][-6:]
    rh = data["relativehumidity_2m"][-6:]
    ps = data["surface_pressure"][-6:]

    # IMPORTANT: must match training feature order
# ['rain', 'temp', 'wind', 'humidity', 'pressure']
    X = np.stack([rain, temp, wind, rh, ps], axis=1)

    return X.astype("float32")
