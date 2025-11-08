from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date


# Weather Models
class WeatherResponse(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    precipitation: float = Field(..., description="Precipitation in mm")


# Crop Recommendation Models
class CropRecommendationRequest(BaseModel):
    N: float = Field(..., description="Nitrogen content in soil")
    P: float = Field(..., description="Phosphorus content in soil")
    K: float = Field(..., description="Potassium content in soil")
    temperature: float = Field(..., description="Temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    ph: float = Field(..., description="Soil pH value")
    rainfall: float = Field(..., description="Rainfall in mm")


class CropStatus(BaseModel):
    status: str
    chance: float
    summary: str


class CropRecommendationResponse(BaseModel):
    recommended_crops: List[str]
    selected_crop: CropStatus
    summary: str


# Crop Yield Prediction Models
class YieldPredictionRequest(BaseModel):
    state: str
    district: str
    year: int
    crop: str
    season: str
    sowing_date: str
    duration_days: int
    soil_oc_pct: float = Field(..., description="Soil organic carbon percentage")
    soil_ph: float = Field(..., description="Soil pH value")
    n_kg_ha: float = Field(..., description="Nitrogen in kg/ha")
    p2o5_kg_ha: float = Field(..., description="Phosphorus in kg/ha")
    k2o_kg_ha: float = Field(..., description="Potassium in kg/ha")
    irrigation_mm: float = Field(..., description="Irrigation in mm")
    rain_mm_season: float = Field(..., description="Seasonal rainfall in mm")
    gdd_base10: float = Field(..., description="Growing degree days")
    hot_days_gt35: int = Field(..., description="Days with temp > 35°C")
    cold_days_lt10: int = Field(..., description="Days with temp < 10°C")
    predict: str = Field(default="yield_kg_ha", description="Prediction type")


class YieldPredictionResponse(BaseModel):
    yield_kg_ha: float
    predicted_yield_tons: float
    confidence: float
    factors: dict
    recommendations: List[str]


# Government Schemes Models
class Scheme(BaseModel):
    id: str
    name: str
    description: str
    eligibility: str
    benefits: str
    application_link: Optional[str] = None
    deadline: Optional[str] = None


class GovSchemesResponse(BaseModel):
    schemes: List[Scheme]
    total_count: int


# LLM Assistant Models
class LLMRequest(BaseModel):
    query: str
    target_language: str = Field(default="en", description="Target language code (en, hi, ta, te, etc.)")
    user_id: Optional[str] = None


class LLMResponse(BaseModel):
    response: str
    language: str
    audio_url: Optional[str] = None


class BotLLMRequest(BaseModel):
    query: str
    target_language: str = Field(default="en", description="Target language code")
    input_type: str = Field(default="text", description="text or audio")
    output_type: str = Field(default="text", description="text or audio")
    user_id: Optional[str] = None


class BotLLMResponse(BaseModel):
    response_text: str
    language: str
    audio_url: Optional[str] = None
    rag_sources: Optional[List[str]] = None
