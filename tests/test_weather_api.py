import requests


def get_weather(latitude: float, longitude: float) -> dict:
    """
    Fetch current weather data from Open-Meteo API.

    Args:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location

    Returns:
        dict: Current weather data
    """

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&current="
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation,"
        "pressure_msl,"
        "wind_speed_10m,"
        "weather_code"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        return data.get("current", {})

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return {}


if __name__ == "__main__":
    lat = float(input("Enter Latitude: "))
    lon = float(input("Enter Longitude: "))

    weather = get_weather(lat, lon)

    if weather:
        print("\nCurrent Weather Data:")
        print("-" * 30)

        for key, value in weather.items():
            print(f"{key}: {value}")

    else:
        print("Failed to fetch weather data.")