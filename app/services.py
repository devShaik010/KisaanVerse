import httpx
from typing import Optional, List, Dict
import pickle
import os
from pathlib import Path


# Load ML Models for Crop Recommendation
MODEL_DIR = Path(__file__).parent / "models" / "Crop_predict"
_crop_model = None
_label_encoder = None
_scaler = None


def load_crop_model():
    """Load CatBoost crop recommendation model, scaler, and label encoder"""
    global _crop_model, _label_encoder, _scaler
    
    if _crop_model is None:
        try:
            import json
            
            # Load the CatBoost model from cbm file
            print("🔄 Loading CatBoost model from cbm file...")
            from catboost import CatBoostClassifier
            model_path = MODEL_DIR / "catboost_crop_rec.cbm"
            _crop_model = CatBoostClassifier()
            _crop_model.load_model(str(model_path))
            
            # Try to load scaler, but don't fail if it has issues
            scaler_path = MODEL_DIR / "scaler.pkl"
            if scaler_path.exists():
                try:
                    print("🔄 Attempting to load feature scaler...")
                    with open(scaler_path, 'rb') as f:
                        _scaler = pickle.load(f)
                    print("✅ Feature scaler loaded")
                except Exception as e:
                    print(f"⚠️ Could not load scaler (pickle compatibility issue): {e}")
                    print("⚠️ Continuing without scaler - model should work without it")
                    _scaler = None
            else:
                print("⚠️ No scaler found, using raw features")
                _scaler = None
            
            # Load metadata to get crop classes
            metadata_path = MODEL_DIR / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                _label_encoder = metadata['classes']
                
            print(f"✅ Crop recommendation model loaded successfully")
            print(f"✅ Model accuracy: {metadata.get('test_accuracy', 'N/A')}")
            print(f"✅ Number of crops: {len(_label_encoder)}")
            
        except Exception as e:
            print(f"❌ Error loading crop model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return _crop_model, _label_encoder, _scaler


# Weather Service
async def get_weather_data(lat: float, lon: float):
    """Get weather data using Open-Meteo API (free, no API key needed)"""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "auto"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        data = response.json()
    
    current = data["current"]
    return {
        "temperature": current["temperature_2m"],
        "humidity": current["relative_humidity_2m"],
        "precipitation": current["precipitation"]
    }


# Geocoding Service
async def get_coordinates(location: str):
    """Convert location name to coordinates using Nominatim (free, no API key)"""
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": location,
        "format": "json",
        "limit": 1
    }
    headers = {
        "User-Agent": "KisaanVerse/1.0"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params, headers=headers)
        data = response.json()
    
    if not data:
        raise ValueError(f"Location '{location}' not found")
    
    return {
        "lat": float(data[0]["lat"]),
        "lon": float(data[0]["lon"])
    }


# Crop Recommendation Service
async def recommend_crops(N: float, P: float, K: float, temperature: float, 
                         humidity: float, ph: float, rainfall: float) -> Dict:
    """
    Predict suitable crops from soil and climate inputs using CatBoost.
    
    Args:
        N: Nitrogen content in soil
        P: Phosphorus content in soil
        K: Potassium content in soil
        temperature: Temperature in Celsius
        humidity: Humidity percentage
        ph: Soil pH value
        rainfall: Rainfall in mm
    
    Returns:
        Dictionary with recommended crops, selected crop details, and summary
    """
    try:
        import numpy as np
        
        # Load model and scaler
        model, crop_classes, scaler = load_crop_model()
        
        # Prepare input data (order must match training: N, P, K, temperature, humidity, ph, rainfall)
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # Apply scaling if scaler exists
        if scaler is not None:
            print(f"🔍 Scaling input features...")
            input_data = scaler.transform(input_data)
        
        # Debug: Print input data
        print(f"🔍 Input data: N={N}, P={P}, K={K}, temp={temperature}, humidity={humidity}, ph={ph}, rain={rainfall}")
        print(f"🔍 Input array shape: {input_data.shape}")
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        
        # Debug: Print top 5 probabilities
        print(f"🔍 Probabilities shape: {probabilities.shape}")
        print(f"🔍 Top 5 probs: {sorted(probabilities, reverse=True)[:5]}")
        
        # Check if model is broken (all predictions same)
        if np.std(probabilities) < 0.01:
            print("⚠️ WARNING: Model appears broken (all probabilities similar)")
            print("⚠️ Using rule-based fallback system")
            return _rule_based_recommendation(N, P, K, temperature, humidity, ph, rainfall)
        
        # Get all crop predictions with probabilities
        crop_predictions = []
        for idx, prob in enumerate(probabilities):
            crop_name = crop_classes[idx]  # Use classes list directly
            crop_predictions.append({
                'crop': crop_name,
                'probability': float(prob)
            })
        
        # Sort by probability (descending)
        crop_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Get top 3 crops
        top_3_crops = crop_predictions[:3]
        recommended_crops = [crop['crop'] for crop in top_3_crops]
        
        # Selected crop (highest probability)
        best_crop = top_3_crops[0]
        best_prob = best_crop['probability']
        
        # Determine status based on probability
        if best_prob >= 0.8:
            status = "highly_suitable"
            status_text = "highly suitable"
        elif best_prob >= 0.6:
            status = "suitable"
            status_text = "suitable"
        elif best_prob >= 0.4:
            status = "moderately_suitable"
            status_text = "moderately suitable"
        else:
            status = "low_suitability"
            status_text = "has low suitability"
        
        # Generate summary for selected crop
        selected_summary = (
            f"{best_crop['crop'].capitalize()} is {status_text} for the given soil and climate conditions. "
            f"With {best_prob*100:.1f}% confidence, this crop should perform well based on "
            f"N={N}, P={P}, K={K}, temperature={temperature}°C, humidity={humidity}%, "
            f"pH={ph}, and rainfall={rainfall}mm."
        )
        
        # Generate overall summary
        summary_parts = [
            f"{crop['crop']} ({crop['probability']*100:.0f}%)" 
            for crop in top_3_crops
        ]
        overall_summary = f"Top 3 recommended crops: {', '.join(summary_parts)}"
        
        return {
            'recommended_crops': recommended_crops,
            'selected_crop': {
                'status': status,
                'chance': round(best_prob, 2),
                'summary': selected_summary
            },
            'summary': overall_summary
        }
        
    except Exception as e:
        raise RuntimeError(f"Error in crop recommendation: {str(e)}")


# Yield Prediction Service
async def predict_crop_yield(request_data: dict):
    """Implement ML model for yield prediction"""
    raise NotImplementedError("Yield prediction service not implemented yet")


# Government Schemes Service
async def fetch_government_schemes():
    """Implement web scraping/API for government schemes"""
    raise NotImplementedError("Government schemes service not implemented yet")


# LLM Service
async def generate_llm_response(query: str, language: str):
    """Implement GenAI/LLM logic"""
    raise NotImplementedError("LLM service not implemented yet")


# RAG Bot Service
async def generate_rag_response(query: str, language: str, input_type: str, output_type: str):
    """Implement RAG system with voice support"""
    raise NotImplementedError("RAG bot service not implemented yet")
