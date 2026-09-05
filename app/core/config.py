import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)  # reloaded with Groq key

class Settings(BaseModel):
    PROJECT_NAME: str = "SynapseOS"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # LLM API Keys & Model Configurations
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
    
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Production Medical Prescription OCR (OpenRouter Free Vision Strategy)
    OPENROUTER_PRIMARY_MODEL: str = os.getenv("OPENROUTER_PRIMARY_MODEL", "google/gemma-4-31b-it:free")
    OPENROUTER_SECONDARY_MODEL: str = os.getenv("OPENROUTER_SECONDARY_MODEL", "google/gemma-4-26b-a4b-it:free")
    OPENROUTER_TERTIARY_MODEL: str = os.getenv("OPENROUTER_TERTIARY_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")
    OPENROUTER_TIMEOUT_MS: int = int(os.getenv("OPENROUTER_TIMEOUT_MS", "45000"))
    MAX_PRESCRIPTION_IMAGE_MB: int = int(os.getenv("MAX_PRESCRIPTION_IMAGE_MB", "10"))
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "2048"))
    OCR_SECOND_PASS_ENABLED: bool = os.getenv("OCR_SECOND_PASS_ENABLED", "false").lower() in ("true", "1", "yes")
    OPENROUTER_REFERER: str = os.getenv("OPENROUTER_REFERER", "https://synapseos.health")
    OPENROUTER_APP_TITLE: str = os.getenv("OPENROUTER_APP_TITLE", "SynapseOS Medical OCR")

    # FractureNet YOLOv8 Detection Service (Remote Hugging Face Space / Fast Inference)
    FRACTURE_API_URL: str = os.getenv("FRACTURE_API_URL", "https://yamxxx1-my-fastapi-app.hf.space")
    
    # Meta Official WhatsApp Cloud API (Graph API) Settings
    WHATSAPP_CLOUD_API_TOKEN: str = os.getenv("WHATSAPP_CLOUD_API_TOKEN", "")
    WHATSAPP_PHONE_NUMBER_ID: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    WHATSAPP_BUSINESS_ACCOUNT_ID: str = os.getenv("WHATSAPP_BUSINESS_ACCOUNT_ID", "")
    WHATSAPP_WEBHOOK_VERIFY_TOKEN: str = os.getenv("WHATSAPP_WEBHOOK_VERIFY_TOKEN", "sanjeevni_secret_token_123")
    WHATSAPP_API_VERSION: str = os.getenv("WHATSAPP_API_VERSION", "v20.0")


    
    # Twilio SMS & Model Backend Settings
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    TWILIO_MODEL_BACKEND_URL: str = os.getenv("TWILIO_MODEL_BACKEND_URL", "https://yamxxx1-my-fastapi-app.hf.space")

    # Blockchain / IPFS (Pinata) settings
    PINATA_JWT: str = os.getenv("PINATA_JWT", "")
    PINATA_GATEWAY_URL: str = os.getenv("PINATA_GATEWAY_URL", "https://gateway.pinata.cloud/ipfs")
    BLOCKCHAIN_RPC_URL: str = os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")
    CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")

settings = Settings()
