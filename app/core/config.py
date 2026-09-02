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
    
    # Meta Official WhatsApp Cloud API (Graph API) Settings
    WHATSAPP_CLOUD_API_TOKEN: str = os.getenv("WHATSAPP_CLOUD_API_TOKEN", "")
    WHATSAPP_PHONE_NUMBER_ID: str = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    WHATSAPP_BUSINESS_ACCOUNT_ID: str = os.getenv("WHATSAPP_BUSINESS_ACCOUNT_ID", "")
    WHATSAPP_WEBHOOK_VERIFY_TOKEN: str = os.getenv("WHATSAPP_WEBHOOK_VERIFY_TOKEN", "sanjeevni_secret_token_123")
    WHATSAPP_API_VERSION: str = os.getenv("WHATSAPP_API_VERSION", "v20.0")


    
    # Twilio SMS Settings
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    TWILIO_PHONE_NUMBER: str = os.getenv("TWILIO_PHONE_NUMBER", "")

    # Blockchain / IPFS (Pinata) settings
    PINATA_JWT: str = os.getenv("PINATA_JWT", "")
    PINATA_GATEWAY_URL: str = os.getenv("PINATA_GATEWAY_URL", "https://gateway.pinata.cloud/ipfs")
    BLOCKCHAIN_RPC_URL: str = os.getenv("BLOCKCHAIN_RPC_URL", "http://127.0.0.1:8545")
    CONTRACT_ADDRESS: str = os.getenv("CONTRACT_ADDRESS", "")

settings = Settings()
