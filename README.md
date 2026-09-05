# 🐍 SynapseOS — Backend API & Architecture

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker)](https://docker.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white)](https://pydantic.dev)

Comprehensive developer guide for setting up, running, testing, and extending the **SynapseOS Multi-Agent FastAPI Backend** for the SMART VIThackathon(SVH)-2026.

---

## 🚀 Quickstart

### 1. Install Dependencies
```powershell
python -m pip install -r backend/requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
cp .env.example .env
```
*(Optional: Add `GROQ_API_KEY`, `OPENROUTER_API_KEY`, or `GEMINI_API_KEY` for live LLM reasoning).*

### 3. Start the FastAPI Server
```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🔗 Documentation & Interactive Endpoints

| Resource | URL | Description |
| :--- | :--- | :--- |
| **Root Health Check** | [`http://127.0.0.1:8000/`](http://127.0.0.1:8000/) | Verify API status |
| **Interactive Swagger UI** | [`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs) | Test API endpoints live |
| **ReDoc API Documentation** | [`http://127.0.0.1:8000/redoc`](http://127.0.0.1:8000/redoc) | Detailed schema reference |

---

## 📋 Comprehensive API Reference

| Endpoint | Method | Tag | Description |
| :--- | :---: | :--- | :--- |
| `/api/orchestrate` | `POST` | Agent Swarm | Runs the full multi-agent DAG workflow with live trace badges. |
| `/api/triage` | `POST` | Clinical Intelligence | Categorizes symptoms into Emergency, Doctor Consult, or Home Care. |
| `/api/drugs/check` | `POST` | Clinical Intelligence | NIH RxNav medication lookup and drug-drug interaction analyzer. |
| `/api/scans/analyze` | `POST` | Vision AI | FractureNet YOLOv8 bone detection, MONAI chest radiographs, & TrOCR. |
| `/api/digital-twin/simulate` | `POST` | Digital Health Twin | 10-year longitudinal multi-organ trajectory simulation. |
| `/api/digital-twin/baseline` | `GET` | Digital Health Twin | Baseline 3D organ vitality scores (Heart, Kidneys, Liver, Pancreas, Lungs). |
| `/api/diagnostics/risk-score` | `POST` | Clinical ML | Quantitative Framingham CVD, ADA Diabetes, CKD eGFR, & FIB-4 Liver. |
| `/api/abdm/generate-id` | `GET` | ABDM & Schemes | Generates mock Indian 14-digit ABHA Health ID and address. |
| `/api/abdm/schemes` | `GET` | ABDM & Schemes | Returns Indian Government health schemes (PM-JAY, Jan Aushadhi, Tele-MANAS). |
| `/api/reports/generate-pdf` | `POST` | Health Records | Generates verifiable health summary PDF with SHA-256 blockchain QR. |
| `/api/sos/dispatch` | `POST` | Emergency SOS | Dispatches 1-click Emergency SOS alert packet with GPS navigation. |
| `/api/whatsapp/webhook` | `GET` | Omnichannel | Official Meta Webhook verification challenge handshake. |
| `/api/whatsapp/webhook` | `POST` | Omnichannel | Official Meta WhatsApp Cloud API inbound webhook handler. |
| `/api/whatsapp/simulate` | `POST` | Omnichannel | Instant simulation testing of WhatsApp messages and scan uploads. |
| `/api/sms/webhook` | `POST` | Omnichannel 2G SMS | Twilio 2-way inbound SMS webhook with XML TwiML responses. |
| `/api/sms/send` | `POST` | Omnichannel 2G SMS | Dispatches outbound SMS messages via Twilio REST API. |
| `/api/sms/inbound` | `POST` | Omnichannel 2G SMS | 2G Plain-Text SMS parser for basic keypad phones. |
| `/api/ipfs/pin-json` | `POST` | Decentralized IPFS | Pins clinical summaries & FHIR records to Pinata IPFS. |
| `/api/fhir/bundle` | `GET` | EHR & FHIR R4 | Generates official HL7 FHIR R4 Bundle (Patient, Observation, Condition). |
| `/api/retrieval/search` | `GET` | Hybrid RAG | Searches WHO/ICMR 23-guideline corpus and Wikipedia medical REST API. |
| `/api/appointments/doctors` | `GET` | Logistics | Lists available PM-JAY empanelled doctors by specialty. |
| `/api/appointments/schedule` | `POST` | Logistics | Books doctor consultation slot and generates digital booking token. |
| `/api/i18n/translate` | `GET` | Multilingual | Translates clinical messages into 11 Indian regional languages. |

---

## 🧪 Running the Test Suite

SynapseOS comes with a **55-test comprehensive test suite** covering all API endpoints, Meta WhatsApp Cloud API, 2-Way SMS Gateway with Twilio, Pinata IPFS Decentralized Storage, ML risk models, and clinical agents:

```powershell
# Run the entire test suite (55 tests)
python -m pytest backend/tests -v

# Run specific test suites
python -m pytest backend/tests/test_sms_and_pinata_service.py -v
python -m pytest backend/tests/test_whatsapp_service.py -v
python -m pytest backend/tests/test_api_endpoints.py -v
python -m pytest backend/tests/test_clinical_ml_and_agents.py -v
```

---

## 📁 Backend Directory Layout

```
backend/
├── app/
│   ├── agents/            # Specialist AI nodes (Triage, Drug, Scan, Mental, Council, etc.)
│   ├── api/               # FastAPI route definitions (endpoints.py)
│   ├── core/              # Config, State Graph, Session Manager & Safety Gate
│   ├── ml/                # Diagnostics ML & 10-Year Digital Twin Trajectory
│   ├── services/          # SMS (Twilio), Pinata IPFS, Meta WhatsApp, FHIR R4, ABDM, PDF & i18n
│   └── main.py            # FastAPI Application entrypoint & CORS middleware
├── tests/                 # Full 55-test suite (test_sms_and_pinata_service.py, etc.)
├── Final.pt               # FractureNet YOLOv8 bone fracture model weights
└── requirements.txt       # Python dependencies
```

---
<div align="center">

### 🔹 built with love by TEAM, AC-DC FOR SMART VIThackathon(SVH)-2026

</div>
