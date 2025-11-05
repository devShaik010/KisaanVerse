import os

class Settings:
    API_TITLE = "KisaanVerse API"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    ALLOWED_ORIGINS = ["*"]
    
    # Weather & Geocoding (Open-Meteo & Nominatim - free, no API key)
    # Add API keys here when using other services
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
    WEATHERAPI_KEY = os.getenv("WEATHERAPI_KEY", "")


settings = Settings()
