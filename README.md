# SynapseOS — Autonomous AI Multi-Agent Health Platform (Backend API)

Production FastAPI backend powering the SynapseOS Clinical Multi-Agent Swarm, Meta WhatsApp Cloud API webhooks, ABDM health records, and clinical triage.

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your credentials:
```bash
cp .env.example .env
```

Key environment variables:
- `WHATSAPP_CLOUD_API_TOKEN`: Meta WhatsApp Cloud API access token.
- `WHATSAPP_PHONE_NUMBER_ID`: WhatsApp Phone Number ID.
- `WHATSAPP_BUSINESS_ACCOUNT_ID`: WhatsApp Business Account ID.
- `WHATSAPP_WEBHOOK_VERIFY_TOKEN`: Verification token for Meta webhook handshake (default: `sanjeevni_secret_token_123`).
- `GROQ_API_KEY`: Groq API key for LLM reasoning.

### 3. Run Locally
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger docs available at: `http://localhost:8000/docs`.

---

## 🌐 WhatsApp Cloud API Webhook

- **Callback URL**: `https://<YOUR_DOMAIN>/api/whatsapp/webhook`
- **Verify Token**: `sanjeevni_secret_token_123` (or configured `WHATSAPP_WEBHOOK_VERIFY_TOKEN`)
- **Webhook Fields**: `messages`

---

## 🚀 Deployment (Render, Koyeb, Docker)

### Render / Koyeb
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Docker
```bash
docker build -t synapse-backend .
docker run -p 8000:8000 --env-file .env synapse-backend
```
