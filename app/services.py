import httpx
from typing import Optional, List, Dict
import pickle
import os
from pathlib import Path


# Load ML Models for Crop Recommendation
CROP_MODEL_DIR = Path(__file__).parent / "models" / "Crop_predict"
YIELD_MODEL_DIR = Path(__file__).parent / "models" / "Yield_prediction"

_crop_model = None
_label_encoder = None
_scaler = None

_yield_model = None
_yield_preprocessor = None
_yield_metadata = None


def load_crop_model():
    """Load CatBoost crop recommendation model"""
    global _crop_model, _label_encoder, _scaler
    
    if _crop_model is None:
        try:
            print("🔄 Loading CatBoost crop recommendation model...")
            
            # Load the new pickle model
            model_path = CROP_MODEL_DIR / "catboost_crop_recommendation_model.pkl"
            with open(model_path, 'rb') as f:
                _crop_model = pickle.load(f)
            
            # Get classes directly from the model
            _label_encoder = _crop_model.classes_
            _scaler = None  # Not needed with the new model
                
            print(f"✅ Crop recommendation model loaded successfully")
            print(f"✅ Number of crops: {len(_label_encoder)}")
            print(f"✅ Crops: {list(_label_encoder)}")
            
        except Exception as e:
            print(f"❌ Error loading crop model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return _crop_model, _label_encoder, _scaler


def load_yield_model():
    """Load CatBoost yield prediction model"""
    global _yield_model, _yield_preprocessor, _yield_metadata
    
    if _yield_model is None:
        try:
            print("🔄 Loading CatBoost yield prediction model...")
            
            # Load the CatBoost model from cbm file
            from catboost import CatBoostRegressor
            model_path = YIELD_MODEL_DIR / "catboost_crop_yield_model.cbm"
            _yield_model = CatBoostRegressor()
            _yield_model.load_model(str(model_path))
            
            # Store feature names from the model
            _yield_metadata = {
                'feature_names': _yield_model.feature_names_
            }
            
            print(f"✅ Yield prediction model loaded successfully")
            print(f"✅ Number of features: {len(_yield_model.feature_names_)}")
            
        except Exception as e:
            print(f"❌ Error loading yield model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    return _yield_model, None, _yield_metadata


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
        import pandas as pd
        
        # Load model
        model, crop_classes, scaler = load_crop_model()
        
        # Prepare input data as DataFrame (matching training format)
        input_data = pd.DataFrame({
            'N': [N],
            'P': [P],
            'K': [K],
            'temperature': [temperature],
            'humidity': [humidity],
            'ph': [ph],
            'rainfall': [rainfall]
        })
        
        # Debug: Print input data
        print(f"🔍 Input data: N={N}, P={P}, K={K}, temp={temperature}, humidity={humidity}, ph={ph}, rain={rainfall}")
        print(f"🔍 Input DataFrame shape: {input_data.shape}")
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        
        # Debug: Print top 5 probabilities
        print(f"🔍 Probabilities shape: {probabilities.shape}")
        print(f"🔍 Top 5 probs: {sorted(probabilities, reverse=True)[:5]}")
        
        # Check if model is broken (all predictions same)
        if np.std(probabilities) < 0.01:
            print("⚠️ WARNING: Model appears broken (all probabilities similar)")
            raise RuntimeError("Model is not responding to input variations. Please retrain the model.")
        
        # Get all crop predictions with probabilities
        crop_predictions = []
        for idx, prob in enumerate(probabilities):
            crop_name = crop_classes[idx]
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
async def predict_crop_yield(Area: float, Annual_Rainfall: float, Fertilizer: float,
                            Pesticide: float, Crop: str, Season: str, State: str) -> Dict:
    """
    Predict crop yield using CatBoost model with one-hot encoding.
    
    Args:
        Area: Area in hectares
        Annual_Rainfall: Annual rainfall in mm
        Fertilizer: Fertilizer amount in kg
        Pesticide: Pesticide amount in kg
        Crop: Crop name
        Season: Season name
        State: State name
    
    Returns:
        Dictionary with predicted yield and recommendations
    """
    try:
        import pandas as pd
        import numpy as np
        
        # Load model
        model, _, metadata = load_yield_model()
        
        # Get all feature names from the model
        all_features = metadata['feature_names']
        
        # Create base input with numeric features
        test_input_data = {
            'Area': [Area],
            'Annual_Rainfall': [Annual_Rainfall],
            'Fertilizer': [Fertilizer],
            'Pesticide': [Pesticide]
        }
        
        # Initialize all encoded columns with 0
        for col in all_features:
            if col not in test_input_data:
                test_input_data[col] = 0
        
        # Set the one-hot encoded columns for Crop, Season, State
        crop_col = f'Crop_{Crop}'
        season_col = f'Season_{Season}'
        state_col = f'State_{State}'
        
        if crop_col in all_features:
            test_input_data[crop_col] = 1
        if season_col in all_features:
            test_input_data[season_col] = 1
        if state_col in all_features:
            test_input_data[state_col] = 1
        
        # Create DataFrame and ensure column order matches training
        test_input = pd.DataFrame(test_input_data)
        test_input = test_input[all_features]
        
        print(f"🔍 Yield Input: Area={Area}, Rainfall={Annual_Rainfall}mm, Crop={Crop}, Season={Season}, State={State}")
        print(f"🔍 Input shape: {test_input.shape}")
        
        # Predict
        predicted_yield = model.predict(test_input)[0]
        
        print(f"🔍 Predicted yield: {predicted_yield:.4f} tons/hectare")
        
        # Calculate total production
        total_production = predicted_yield * Area
        area_in_acres = Area * 2.47105  # Convert hectares to acres
        
        print(f"🔍 Total production: {total_production:.2f} tons")
        print(f"🔍 Area: {Area} hectares ({area_in_acres:.2f} acres)")
        
        # Determine confidence level
        if predicted_yield > 0.5:
            confidence = "High"
        elif predicted_yield > 0.1:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Generate recommendations
        recommendations = []
        
        # Area-based recommendations
        if Area < 1:
            recommendations.append(f"Small plot of {area_in_acres:.2f} acres. Ideal for intensive farming and high-value crops.")
        elif Area < 5:
            recommendations.append(f"Medium-sized farm of {area_in_acres:.2f} acres. Good for diversified cropping.")
        elif Area < 20:
            recommendations.append(f"Large farm of {area_in_acres:.2f} acres. Consider mechanization for efficiency.")
        else:
            recommendations.append(f"Very large farm of {area_in_acres:.2f} acres. Mechanization highly recommended.")
        
        # Production-based recommendations
        if total_production < 10:
            recommendations.append(f"Expected total production: {total_production:.2f} tons. Consider value addition for better returns.")
        elif total_production < 100:
            recommendations.append(f"Expected total production: {total_production:.2f} tons. Plan storage and marketing strategies.")
        else:
            recommendations.append(f"Expected total production: {total_production:.2f} tons. Commercial-scale production - ensure supply chain management.")
        
        # Yield-based recommendations
        if predicted_yield < 1.0:
            recommendations.append(f"Predicted yield is relatively low. Consider optimizing fertilizer application.")
            recommendations.append(f"Ensure adequate irrigation based on rainfall of {Annual_Rainfall}mm.")
        elif predicted_yield > 5.0:
            recommendations.append(f"Excellent yield potential! Maintain current farming practices.")
            recommendations.append(f"Monitor for pests and diseases to protect high-value crops.")
        else:
            recommendations.append(f"Expected yield is within normal range for {Crop}.")
        
        # Rainfall-based recommendations
        if Annual_Rainfall < 500:
            recommendations.append("Low rainfall detected. Plan for supplemental irrigation systems.")
        elif Annual_Rainfall > 2000:
            recommendations.append("High rainfall area. Ensure proper drainage to prevent waterlogging.")
        else:
            recommendations.append(f"Moderate rainfall of {Annual_Rainfall}mm. Suitable for {Crop} cultivation.")
        
        return {
            'predicted_yield': round(predicted_yield, 4),
            'total_production': round(total_production, 2),
            'area': Area,
            'area_in_acres': round(area_in_acres, 2),
            'crop': Crop,
            'season': Season,
            'state': State,
            'confidence_level': confidence,
            'recommendations': recommendations
        }
        
    except Exception as e:
        raise RuntimeError(f"Error in yield prediction: {str(e)}")


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
