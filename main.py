from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router

app = FastAPI(
    title="KisaanVerse API",
    description="Agricultural platform API",
    version="1.0.0",
    contact={
        "name": "KisaanVerse Team",
        "url": "https://github.com/devShaik010/KisaanVerse",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/", tags=["Root"])
async def root():
    return {"message": "KisaanVerse API", "version": "1.0.0", "docs": "/docs"}


@app.get("/health", tags=["Root"])
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
