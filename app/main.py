"""
SynapseOS — Main FastAPI Application Entrypoint.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import settings
from backend.app.api.endpoints import router as api_router

from contextlib import asynccontextmanager
from backend.app.agents.scan_agent import get_fracture_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm FractureNet YOLOv8 model in memory
    try:
        get_fracture_model()
    except Exception as e:
        pass
    yield

app = FastAPI(
    title="SynapseOS — Multi-Agent Health Platform API",
    version=settings.VERSION,
    description="Autonomous AI-first Health Operating System powering dual-mode health assistants, clinical ML, digital twin, ABDM, and omnichannel care.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for Next.js frontend (default port 3000) & external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount main API router
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/", tags=["Health Check"])
@app.get("/health", tags=["Health Check"])
@app.get("/api/health", tags=["Health Check"])
async def root():
    return {
        "platform": "SynapseOS",
        "version": settings.VERSION,
        "status": "ONLINE",
        "agents_active": [
            "Deterministic Safety Gate",
            "Orchestrator Agent",
            "Clinical Symptom Triage Agent",
            "National Universal Immunization & U-WIN Vaccine Agent",
            "Rural Preventive Healthcare & Community Education Agent",
            "IDSP Epidemic Outbreak & Early Warning Agent",
            "Pharmacology & RxNav Drug Safety Agent",
            "MONAI Medical Scan & Vision Agent",
            "AI Council & Evidence Grounding Agent (80%+ Accuracy Benchmark)",
            "WHO/Tele-MANAS Mental Health Agent",
            "3D Digital Health Twin Simulation Engine",
            "Ayushman Bharat ABDM / ABHA Service",
            "Omnichannel 2G SMS & Meta Official WhatsApp Cloud API Gateway"
        ],
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
