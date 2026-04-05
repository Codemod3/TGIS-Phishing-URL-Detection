from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, health, history

# Initializing FastAPI application following ARCHITECTURE.md specifications
app = FastAPI(
    title="Phishing URL Detector API",
    version="1.0.0",
    description="Elite phishing detection combining ML and Trust Graph Intelligence (TGIS)"
)

@app.on_event("startup")
def setup_database():
    """Ensures database tables are created on startup without crashing the app."""
    try:
        from api.database import engine, Base
        import api.models
        Base.metadata.create_all(bind=engine)
        print("✅ Database schema synchronized successfully.")
    except Exception as e:
        print(f"⚠️ Database connection failed: {e}")
        print("System will continue without persistence layer.")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Including Routers
app.include_router(predict.router)
app.include_router(health.router)
app.include_router(history.router)

@app.get("/")
def root():
    """Welcome point for the Elite Phishing Detection System."""
    return {
        "app": "Phishing URL Detector API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # Local development server entry point
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
