"""
SynapseOS — Backend Unit Tests.
Tests safety guardrails, drug interaction checking, digital twin, and orchestrator swarm.
"""

import pytest
from backend.app.core.safety_router import evaluate_safety
from backend.app.agents.drug_agent import evaluate_drug_safety
from backend.app.agents.triage_agent import analyze_symptoms
from backend.app.agents.orchestrator import orchestrate_health_request
from backend.app.ml.digital_twin import DigitalTwinInput, compute_baseline_organ_scores
from backend.app.ml.diagnostics import DiagnosticRiskRequest, calculate_clinical_risks
from backend.app.services.abdm_service import generate_abha_id
from backend.app.services.pdf_service import generate_health_summary_pdf


def test_safety_router_crisis_interception():
    # Test suicidal / crisis interception
    res = evaluate_safety("I want to end my life, I can't take it anymore")
    assert not res.is_safe
    assert res.category == "crisis"
    assert "Tele-MANAS" in res.response

    # Test emergency red-flag interception
    res_em = evaluate_safety("I have severe crushing chest pain and difficulty breathing")
    assert not res_em.is_safe
    assert res_em.category == "emergency"
    assert "POTENTIAL MEDICAL EMERGENCY" in res_em.response

    # Test normal safe query
    res_safe = evaluate_safety("What is the standard dosage for vitamin C?")
    assert res_safe.is_safe


@pytest.mark.asyncio
async def test_drug_interaction_detection():
    # Aspirin + Warfarin dangerous pair
    res = await evaluate_drug_safety("Can I take aspirin with warfarin?")
    assert "aspirin" in res["detected_medications"]
    assert "warfarin" in res["detected_medications"]
    assert res["interactions_count"] > 0
    assert res["safe_to_combine"] is False


@pytest.mark.asyncio
async def test_clinical_symptom_triage():
    res = await analyze_symptoms("I have a persistent fever over 102 and abdominal pain")
    assert res["triage_level"] in ("DOCTOR_CONSULT", "EMERGENCY_CARE")
    assert any(term in res["urgency_badge"] for term in ("Doctor Consultation", "Emergency"))


def test_i18n_indian_languages():
    from backend.app.services.i18n_service import translate_clinical_message, get_supported_languages
    langs = get_supported_languages()
    assert len(langs) >= 11
    
    # Test Hindi
    hi_msg = translate_clinical_message("emergency_alert", "hi")
    assert "112" in hi_msg
    
    # Test Tamil
    ta_msg = translate_clinical_message("emergency_alert", "ta")
    assert "112" in ta_msg

    # Test Bengali
    bn_msg = translate_clinical_message("emergency_alert", "bn")
    assert "১১২" in bn_msg or "112" in bn_msg


def test_digital_twin_scores():
    req = DigitalTwinInput(systolic_bp=150, fasting_glucose=140, smoking_status="current")
    scores = compute_baseline_organ_scores(req)
    assert scores["overall_health_score"] < 85
    assert "heart" in scores["organs"]
    assert scores["organs"]["heart"]["score"] < 80


def test_clinical_risk_prediction():
    req = DiagnosticRiskRequest(age=55, systolic_bp=145, total_cholesterol=240, is_smoker=True)
    risks = calculate_clinical_risks(req)
    assert risks["cardiovascular_risk"]["ten_year_probability_percent"] > 10
    assert "renal_health" in risks


def test_abdm_abha_generation():
    abha = generate_abha_id(name="Aarav Sharma", year_of_birth=1992)
    assert abha["status"] == "ACTIVE"
    assert len(abha["abha_number"].split("-")) == 4
    assert "@abdm" in abha["abha_address"]


def test_pdf_report_generation():
    pdf_bytes = generate_health_summary_pdf(patient_name="Test Citizen", abha_id="91-1234-5678-9012")
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


@pytest.mark.asyncio
async def test_full_orchestrator_swarm():
    state = await orchestrate_health_request("I have fever and want to know if paracetamol is safe")
    assert state.safety_cleared is True
    assert len(state.trace) >= 3
    assert state.final_response != ""


def test_fhir_r4_bundle_generation():
    from backend.app.services.fhir_service import build_fhir_r4_bundle
    bundle = build_fhir_r4_bundle("PAT-123", "Rohan Gupta", vitals={"systolic_bp": 120})
    assert bundle["resourceType"] == "Bundle"
    assert bundle["total"] >= 2
    assert bundle["entry"][0]["resource"]["resourceType"] == "Patient"


@pytest.mark.asyncio
async def test_hybrid_clinical_rag():
    from backend.app.agents.retrieval_agent import hybrid_retrieve_clinical_context
    rag = await hybrid_retrieve_clinical_context("diabetes management")
    assert rag["retrieval_sources_count"] > 0
    assert len(rag["who_icmr_guidelines"]) > 0


def test_appointment_booking():
    from backend.app.agents.appointment_agent import book_appointment_slot, find_doctors_by_specialty
    docs = find_doctors_by_specialty("Cardiologist")
    assert len(docs) > 0
    booking = book_appointment_slot("Pooja Verma", docs[0]["doctor_id"], "Tomorrow at 10:30 AM")
    assert booking["status"] == "CONFIRMED"
    assert booking["booking_id"].startswith("APT-")


def test_i18n_translation():
    from backend.app.services.i18n_service import translate_clinical_message
    hindi = translate_clinical_message("emergency_alert", "hi")
    assert "आपातकालीन" in hindi

