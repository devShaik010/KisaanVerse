# KisaanVerse - Developer Setup Guide

FastAPI backend for agricultural platform with AI-powered features.

## Project Setup

### 1. Clone Repository
```powershell
git clone https://github.com/devShaik010/KisaanVerse.git
cd KisaanVerse
```

### 2. Create Virtual Environment
```powershell
python -m venv env
```

### 3. Activate Environment
```powershell
.\env\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Run Application
```powershell
python main.py
```

Access API Documentation: **http://localhost:8000/docs**

## Project Structure

```
KisaanVerse/
├── app/
│   ├── __init__.py
│   ├── config.py       # Configuration & API keys
│   ├── models.py       # Pydantic request/response models
│   ├── routes.py       # API endpoints
│   ├── services.py     # Business logic (add your implementations here)
│   └── models/
│       └── croppredict.pkl  # ML model files
├── main.py             # FastAPI application entry point
├── requirements.txt    # Python dependencies
└── README.md
```

## How to Work

### Adding New Features

1. **Define Models** (`app/models.py`)
   - Add Pydantic models for request/response

2. **Implement Logic** (`app/services.py`)
   - Write your business logic functions
   - Integrate ML models, APIs, databases

3. **Create Endpoints** (`app/routes.py`)
   - Connect routes to services
   - Handle requests and responses

4. **Configure Settings** (`app/config.py`)
   - Add API keys
   - Set environment variables

### Available Endpoints

| Method | Endpoint | Purpose | Status |
|--------|----------|---------|--------|
| GET | `/KV/weather` | Get weather data by location/coordinates | ✅ Working |
| POST | `/kv/predictCrop` | Crop recommendation system | ⚠️ TODO |
| POST | `/kv/yieldPredict` | Crop yield prediction | ⚠️ TODO |
| GET | `/kv/govSchemes` | Government schemes list | ⚠️ TODO |
| POST | `/kv/kvLLM` | AI assistant | ⚠️ TODO |
| POST | `/kv/BotLLM` | RAG-powered voice bot | ⚠️ TODO |

### Testing Endpoints

1. **Using Swagger UI**
   - Open: http://localhost:8000/docs
   - Click on endpoint → "Try it out" → Fill parameters → Execute

2. **Using cURL**
   ```powershell
   # Weather by location
   curl "http://localhost:8000/KV/weather?location=Bangalore"
   
   # Weather by coordinates
   curl "http://localhost:8000/KV/weather?lat=12.9716&lon=77.5946"
   ```

## Tech Stack

- **Backend**: FastAPI, Pydantic, Uvicorn
- **HTTP Client**: httpx (for external APIs)
- **Weather**: Open-Meteo (free, no API key required)
- **Geocoding**: Nominatim (free, no API key required)

## Development Tips

- Hot reload is enabled - code changes auto-restart server
- Check `/health` endpoint to verify server is running
- All endpoints return JSON responses
- Use Swagger UI for interactive API testing

## Next Steps

1. Implement crop recommendation ML model in `services.py`
2. Add yield prediction logic
3. Integrate government schemes web scraping
4. Connect GenAI/LLM for AI assistant
5. Add database for user data
6. Implement authentication

---

**License**: MIT
