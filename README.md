# KisaanVerse 🌾

> Enhancing Farmer Productivity through Innovative Technology Solutions

KisaanVerse is a comprehensive mobile application designed to empower farmers with AI-driven insights, real-time weather monitoring, crop recommendations, and government scheme information in their regional language.

## Features

- **Crop Recommendation System** - Get personalized crop suggestions based on location, soil, and season
- **Crop Yield Prediction** - Predict crop yields with ML-powered analytics and receive timely alerts
- **Weather Monitoring** - Real-time weather status with temperature, humidity, and precipitation data
- **Government Schemes Dashboard** - Stay updated with relevant agricultural schemes and subsidies
- **Multilingual Voice & Chat Assistant** - AI-powered assistant with regional language support

## Tech Stack

### Backend
- FastAPI / Flask
- GenAI & ML Models
- Langchain & RAG System
- ChromaDB & FaissIndex
- Crawl4AI for web scraping
- Docker & Cloud deployment

### Mobile App
- Flutter (Dart/Java)
- REST API Integration
- Android SDK
- Real-time notifications

### UI/UX
- Figma & Canva
- Pinterest-inspired designs
- Excalidraw

## API Endpoints

### Weather Status
```
GET /KV/weather
Response: { temperature, humidity, precipitation }
```

### Crop Recommendation
```
POST /kv/predictCrop
Request: { state, crop, area, fertilizer, pesticide, season }
Response: { recommended_crops, selected_crop_status, ai_summary }
```

### Yield Prediction
```
POST /kv/yieldPredict
Request: { state, district, year, crop, season, soil_data, weather_data }
Response: { yield_kg_ha, predictions }
```

### Government Schemes
```
GET /kv/govSchemes
Response: { schemes_list }
```

### AI Assistant
```
POST /kv/kvLLM
POST /kv/BotLLM
Features: Text/Voice input & output, Regional language support
```

## User Flow

1. **Intro Screen** → Language Selection
2. **Registration** → Name, Location, Phone
3. **Intro Screens** → Feature walkthrough (in selected language)
4. **Home Screen** → Access to 4 main features + weather status
5. Language can be changed anytime via settings icon

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/KisaanVerse.git

# Navigate to project directory
cd KisaanVerse

# Install dependencies
# (Add specific installation commands based on your setup)
```

## Development

- **IDE**: VS Code
- **Mobile Testing**: Android Emulator
- **Flutter SDK**: Required
- **Android SDK**: Required

## Keywords

`agriculture` `genai` `ml` `flutter` `regional-language` `crop-prediction` `weather-monitoring` `voice-assistant` `rag-system` `fastapi`

---

**License**: MIT (or your preferred license)

**Contributors**: Welcome! Please read our contributing guidelines before submitting PRs.
