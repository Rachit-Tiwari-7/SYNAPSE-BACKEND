"""
SynapseOS — tests/test_api_endpoints.py
Comprehensive integration tests for all FastAPI REST endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Tests the root health and service information endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ONLINE"
    assert data["platform"] == "SynapseOS"
    assert "agents_active" in data
    assert len(data["agents_active"]) > 5


def test_orchestrate_endpoint():
    """Tests the full multi-agent orchestrator endpoint."""
    payload = {
        "message": "I have headache and fever, can I take paracetamol?",
        "channel": "web",
        "user_id": "test_patient"
    }
    response = client.post("/api/orchestrate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "final_response" in data
    assert "trace" in data
    assert len(data["trace"]) > 0


def test_triage_endpoint():
    """Tests the clinical symptom triage endpoint."""
    payload = {"symptoms": "Mild sore throat, runny nose and sneezing"}
    response = client.post("/api/triage", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["triage_level"] == "HOME_CARE"
    assert "urgency_badge" in data
    assert "recommended_action" in data


def test_drugs_check_endpoint():
    """Tests the NIH RxNav drug-drug interaction checker endpoint."""
    payload = {"query_or_meds": "Can I take aspirin with warfarin?"}
    response = client.post("/api/drugs/check", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "aspirin" in data["detected_medications"]
    assert "warfarin" in data["detected_medications"]
    assert data["interactions_count"] > 0
    assert data["safe_to_combine"] is False


def test_scans_analyze_endpoint():
    """Tests the medical vision analysis endpoint for chest, fracture, and prescription."""
    # 1. Chest X-Ray
    resp_chest = client.post("/api/scans/analyze", json={"image_type": "chest_xray"})
    assert resp_chest.status_code == 200
    data_chest = resp_chest.json()
    assert data_chest["modality"] == "chest_xray"
    assert "ai_diagnosis_summary" in data_chest

    # 2. Bone Fracture (Default & Upload with HF Space mock)
    resp_bone = client.post("/api/scans/analyze", json={"image_type": "bone_fracture"})
    assert resp_bone.status_code == 200
    data_bone = resp_bone.json()
    assert data_bone["modality"] == "bone_fracture"

    # User upload testing remote HF Space response
    import io
    import base64
    from PIL import Image

    test_img = Image.new("RGB", (200, 200), color=(200, 200, 200))
    buf = io.BytesIO()
    test_img.save(buf, format="JPEG")
    valid_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    mock_hf_response = {
        "detection_id": "test-uuid-123",
        "message": "Detection completed successfully",
        "result_image": "/results/test_result.jpg",
        "explanation_image": "/results/explanations/test_exp.jpg",
        "gradcam_image": "/results/gradcam/test_grad.jpg",
        "detections": [
            {
                "id": 0,
                "class": "fracture",
                "confidence": 0.88,
                "box": {"x1": 50, "y1": 60, "x2": 150, "y2": 180}
            }
        ]
    }
    with patch("backend.app.agents.scan_agent.httpx.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_hf_response
        mock_instance.post.return_value = mock_resp
        mock_client_cls.return_value = mock_instance

        resp_bone_upload = client.post(
            "/api/scans/analyze",
            json={
                "image_type": "bone_fracture",
                "filename": "arm_xray.jpg",
                "image_base64": valid_b64
            }
        )
        assert resp_bone_upload.status_code == 200
        upload_data = resp_bone_upload.json()
        assert upload_data["modality"] == "bone_fracture"
        assert len(upload_data["visual_bounding_boxes"]) == 1
        assert "Fracture: Fracture" in upload_data["visual_bounding_boxes"][0]["label"]
        assert "remote_result_image" in upload_data
        assert "https://yamxxx1-my-fastapi-app.hf.space/results/test_result.jpg" in upload_data["remote_result_image"]

    # 3. Prescription
    resp_rx = client.post("/api/scans/analyze", json={"image_type": "prescription"})
    assert resp_rx.status_code == 200
    data_rx = resp_rx.json()
    assert data_rx["modality"] == "prescription"


def test_digital_twin_endpoints():
    """Tests the 10-year digital health twin trajectory and baseline endpoints."""
    # Baseline
    resp_base = client.get("/api/digital-twin/baseline")
    assert resp_base.status_code == 200
    base_data = resp_base.json()
    assert "overall_health_score" in base_data
    assert "organs" in base_data
    assert "heart" in base_data["organs"]

    # 10-Year Simulation
    sim_input = {
        "age": 45,
        "gender": "male",
        "systolic_bp": 135,
        "fasting_glucose": 110,
        "proposed_interventions": ["exercise", "diet"]
    }
    resp_sim = client.post("/api/digital-twin/simulate", json=sim_input)
    assert resp_sim.status_code == 200
    sim_data = resp_sim.json()
    assert "ten_year_projections" in sim_data
    assert "baseline" in sim_data
    assert "recommendations" in sim_data


def test_diagnostic_risk_score_endpoint():
    """Tests the quantitative clinical risk scoring endpoint (Framingham, ADA, CKD, FIB-4)."""
    payload = {
        "age": 52,
        "gender": "male",
        "systolic_bp": 145.0,
        "total_cholesterol": 230.0,
        "hdl_cholesterol": 38.0,
        "fasting_glucose": 130.0,
        "hba1c": 6.8,
        "creatinine": 1.4,
        "ast": 35.0,
        "alt": 42.0,
        "platelet_count": 210.0,
        "is_smoker": True,
        "is_diabetic": True
    }
    response = client.post("/api/diagnostics/risk-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "cardiovascular_risk" in data
    assert "diabetes_risk" in data
    assert "renal_health" in data
    assert "hepatic_index" in data


def test_abdm_endpoints():
    """Tests the ABHA ID generation and Ayushman Bharat schemes endpoints."""
    resp_id = client.get("/api/abdm/generate-id?name=Siddharth+Sharma&year_of_birth=1990&state_code=DL")
    assert resp_id.status_code == 200
    id_data = resp_id.json()
    assert "abha_number" in id_data
    assert "abha_address" in id_data
    assert id_data["pm_jay_eligible"] is True

    resp_schemes = client.get("/api/abdm/schemes")
    assert resp_schemes.status_code == 200
    schemes_data = resp_schemes.json()
    assert "pmjay" in schemes_data
    assert "jan_aushadhi" in schemes_data


def test_reports_generate_pdf_endpoint():
    """Tests clinical PDF health summary report generation."""
    payload = {
        "patient_name": "Siddharth Sharma",
        "abha_id": "91-5829-3910-4821",
        "triage_summary": "Routine checkup; mild elevated BP.",
        "vital_signs": {"Blood Pressure": "130/85 mmHg", "Blood Glucose": "98 mg/dL"},
        "medications": [{"name": "Amlodipine 5mg", "dosage": "1 OD", "duration": "30 Days"}]
    }
    response = client.post("/api/reports/generate-pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF-")


def test_emergency_sos_dispatch_endpoint():
    """Tests the 1-click Emergency SOS dispatch endpoint."""
    payload = {
        "emergency_contact": "+919876543210",
        "patient_name": "Siddharth Sharma",
        "location_coords": "28.6139,77.2090",
        "blood_group": "O+",
        "critical_symptoms": "Severe acute chest tightness"
    }
    response = client.post("/api/sos/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("SOS_DISPATCHED", "SOS_QUEUED")
    assert "112 (National Emergency)" in data["emergency_services_reference"]


def test_whatsapp_endpoints():
    """Tests the Meta WhatsApp Cloud API verification, inbound webhook, and simulation endpoints."""
    # 1. Meta Webhook GET Handshake Verification
    resp_verify = client.get(
        "/api/whatsapp/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.challenge": "1158201444",
            "hub.verify_token": "sanjeevni_secret_token_123"
        }
    )
    assert resp_verify.status_code == 200
    assert resp_verify.text == "1158201444"

    # 2. Meta Official Format Webhook POST
    meta_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "100000000000000",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "15550234567", "phone_number_id": "100000000000000"},
                    "contacts": [{"profile": {"name": "Sanjeevni User"}, "wa_id": "919876543210"}],
                    "messages": [{
                        "from": "919876543210",
                        "id": "wamid.TEST_META_ID",
                        "timestamp": "1772185000",
                        "type": "text",
                        "text": {"body": "Menu"}
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    resp_webhook = client.post("/api/whatsapp/webhook", json=meta_payload)
    assert resp_webhook.status_code == 200
    assert resp_webhook.json()["status"] == "processed"

    # 3. Simulation Endpoint
    sim_payload = {
        "message": "1 I have slight fever",
        "sender_phone": "919876543210",
        "message_type": "text"
    }
    resp_sim = client.post("/api/whatsapp/simulate", json=sim_payload)
    assert resp_sim.status_code == 200
    assert resp_sim.json()["status"] == "processed"


def test_fhir_bundle_endpoint():
    """Tests the HL7 FHIR R4 Bundle generation endpoint."""
    response = client.get("/api/fhir/bundle?patient_id=PAT-91-4829&name=Siddharth+Sharma")
    assert response.status_code == 200
    bundle = response.json()
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert len(bundle["entry"]) >= 3


def test_retrieval_search_endpoint():
    """Tests the Hybrid Clinical RAG retrieval endpoint."""
    response = client.get("/api/retrieval/search?query=diabetes")
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "who_icmr_guidelines" in data
    assert len(data["who_icmr_guidelines"]) > 0


def test_appointments_endpoints():
    """Tests PM-JAY doctor search and slot booking endpoints."""
    # List Doctors
    resp_docs = client.get("/api/appointments/doctors?specialty=Cardiologist")
    assert resp_docs.status_code == 200
    docs = resp_docs.json()
    assert len(docs) > 0

    # Book Appointment
    resp_book = client.post(
        "/api/appointments/schedule",
        params={
            "patient_name": "Siddharth Sharma",
            "doctor_id": "DOC-AIIMS-101",
            "slot_time": "Tomorrow at 10:30 AM"
        }
    )
    assert resp_book.status_code == 200
    book_data = resp_book.json()
    assert book_data["status"] == "CONFIRMED"
    assert "booking_id" in book_data


def test_i18n_translation_endpoint():
    """Tests the multilingual clinical translation endpoint."""
    response = client.get("/api/i18n/translate?key=emergency_alert&lang=hi")
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "hi"
    assert "translated_text" in data
    assert "112" in data["translated_text"]

    # Language roster
    resp_langs = client.get("/api/i18n/languages")
    assert resp_langs.status_code == 200
    assert resp_langs.json()["count"] >= 11
