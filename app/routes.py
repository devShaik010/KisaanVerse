from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from app.models import (
    WeatherResponse,
    CropRecommendationRequest,
    CropRecommendationResponse,
    YieldPredictionRequest,
    YieldPredictionResponse,
    GovSchemesResponse,
    LLMRequest,
    LLMResponse,
    BotLLMRequest,
    BotLLMResponse
)
from app.services import get_weather_data, get_coordinates

router = APIRouter()


@router.get("/KV/weather", response_model=WeatherResponse, tags=["Weather"])
async def get_weather(
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
    location: Optional[str] = Query(None)
):
    """Get weather data"""
    if location:
        coords = await get_coordinates(location)
        lat, lon = coords["lat"], coords["lon"]
    
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Provide location or lat/lon")
    
    weather = await get_weather_data(lat, lon)
    return WeatherResponse(**weather)


@router.post("/kv/predictCrop", response_model=CropRecommendationResponse, tags=["Crop Recommendation"])
async def predict_crop(request: CropRecommendationRequest):
    """Crop recommendation"""
    pass


@router.post("/kv/yieldPredict", response_model=YieldPredictionResponse, tags=["Yield Prediction"])
async def predict_yield(request: YieldPredictionRequest):
    """Yield prediction"""
    pass


@router.get("/kv/govSchemes", response_model=GovSchemesResponse, tags=["Government Schemes"])
async def get_government_schemes():
    """Government schemes"""
    pass


@router.post("/kv/kvLLM", response_model=LLMResponse, tags=["AI Assistant"])
async def kv_llm_assistant(request: LLMRequest):
    """AI assistant"""
    pass


@router.post("/kv/BotLLM", response_model=BotLLMResponse, tags=["AI Assistant"])
async def bot_llm_assistant(request: BotLLMRequest):
    """Voice bot"""
    pass
