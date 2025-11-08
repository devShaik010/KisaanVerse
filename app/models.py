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
    Area: float = Field(..., description="Area in hectares")
    Annual_Rainfall: float = Field(..., description="Annual rainfall in mm")
    Fertilizer: float = Field(..., description="Fertilizer amount in kg")
    Pesticide: float = Field(..., description="Pesticide amount in kg")
    Crop: str = Field(..., description="Crop name (e.g., Arecanut, Rice, Wheat)")
    Season: str = Field(..., description="Season (Kharif, Rabi, Whole Year, Summer, Winter, Autumn)")
    State: str = Field(..., description="State name (e.g., Assam, Punjab, Maharashtra)")


class YieldPredictionResponse(BaseModel):
    predicted_yield: float = Field(..., description="Predicted yield in tons/hectare")
    total_production: float = Field(..., description="Total production in tons (yield × area)")
    area: float = Field(..., description="Cultivation area in hectares")
    area_in_acres: float = Field(..., description="Cultivation area in acres")
    crop: str
    season: str
    state: str
    confidence_level: str = Field(..., description="High, Medium, or Low")
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
