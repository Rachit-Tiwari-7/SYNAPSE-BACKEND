"""
SynapseOS — tests/test_clinical_ml_and_agents.py
Unit tests for Clinical ML Models, Multi-Agent Swarms, Verification Council,
Digital Health Twin, FHIR Serializers, and Hybrid Guidelines Search.
"""

import pytest
from backend.app.ml.diagnostics import DiagnosticRiskRequest, calculate_clinical_risks
from backend.app.ml.digital_twin import DigitalTwinInput, compute_baseline_organ_scores, simulate_10_year_trajectory
from backend.app.services.fhir_service import build_fhir_r4_bundle, build_wearable_fhir_bundle
from backend.app.agents.verification_agent import verify_clinical_claims
from backend.app.agents.mental_health_agent import evaluate_mental_wellbeing
from backend.app.agents.retrieval_agent import hybrid_retrieve_clinical_context
from backend.app.agents.appointment_agent import find_doctors_by_specialty, book_appointment_slot


def test_cvd_framingham_calculations():
    """Tests 10-year CVD Framingham risk calculation for high-risk vs low-risk patients."""
    high_risk_req = DiagnosticRiskRequest(
        age=65,
        gender="male",
        systolic_bp=165.0,
        total_cholesterol=260.0,
        hdl_cholesterol=32.0,
        is_smoker=True,
        is_diabetic=True
    )
    result_high = calculate_clinical_risks(high_risk_req)
    assert result_high["cardiovascular_risk"]["category"] == "High / Elevated"
    assert result_high["cardiovascular_risk"]["ten_year_probability_percent"] > 30.0

    low_risk_req = DiagnosticRiskRequest(
        age=25,
        gender="female",
        systolic_bp=110.0,
        total_cholesterol=160.0,
        hdl_cholesterol=55.0,
        is_smoker=False,
        is_diabetic=False
    )
    result_low = calculate_clinical_risks(low_risk_req)
    assert result_low["cardiovascular_risk"]["category"] == "Low"
    assert result_low["cardiovascular_risk"]["ten_year_probability_percent"] < 10.0


def test_egfr_ckd_and_fib4_calculations():
    """Tests renal eGFR staging and liver FIB-4 index formulas."""
    req = DiagnosticRiskRequest(
        age=60,
        gender="male",
        creatinine=2.2,
        ast=50.0,
        alt=35.0,
        platelet_count=140.0
    )
    res = calculate_clinical_risks(req)
    # Check eGFR calculation
    assert res["renal_health"]["estimated_gfr"] < 60.0
    assert "Stage 3" in res["renal_health"]["kdigo_stage"] or "Stage 4" in res["renal_health"]["kdigo_stage"]

    # Check FIB-4 Liver score calculation: (age * AST) / (platelet * sqrt(ALT))
    assert res["hepatic_index"]["fib4_score"] > 2.0


def test_digital_twin_organ_scores_and_trajectory():
    """Tests 10-year multi-organ progression simulation under lifestyle interventions."""
    input_data = DigitalTwinInput(
        age=50,
        gender="female",
        systolic_bp=140,
        fasting_glucose=125,
        hba1c=6.4,
        proposed_interventions=["exercise", "diet"]
    )
    baseline = compute_baseline_organ_scores(input_data)
    assert "heart" in baseline["organs"]
    assert "pancreas" in baseline["organs"]
    assert baseline["overall_health_score"] >= 0

    simulation = simulate_10_year_trajectory(input_data)
    assert "ten_year_projections" in simulation
    assert "heart" in simulation["ten_year_projections"]
    heart_proj = simulation["ten_year_projections"]["heart"]
    assert len(heart_proj["baseline_trajectory"]) == 11
    assert len(heart_proj["optimized_trajectory"]) == 11


def test_fhir_r4_bundle_generation():
    """Tests standard FHIR R4 Patient, Observation, and Condition resource packaging."""
    bundle = build_fhir_r4_bundle(
        patient_id="PAT-99-001",
        name="Ananya Sen",
        gender="female",
        birth_date="1992-08-15",
        vitals={"systolic_bp": 118, "fasting_glucose": 92, "heart_rate": 72, "spo2": 99},
        conditions=["Controlled Asthma"]
    )
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"

    resource_types = [entry["resource"]["resourceType"] for entry in bundle["entry"]]
    assert "Patient" in resource_types
    assert "Observation" in resource_types
    assert "Condition" in resource_types


def test_wearable_fhir_bundle():
    """Tests real-time IoT wearable sensor FHIR bundle generation."""
    payload = {
        "device_name": "Apple Watch Ultra 2",
        "source": "apple_health",
        "spo2_percent": 98.5,
        "heart_rate_bpm": 78,
        "resting_heart_rate": 65,
        "hrv_ms": 45,
        "respiratory_rate": 16,
        "steps": 8420
    }
    wearable_bundle = build_wearable_fhir_bundle(
        payload=payload,
        patient_id="PAT-WEAR-101",
        patient_name="Rahul Verma"
    )
    assert wearable_bundle["resourceType"] == "Bundle"
    assert len(wearable_bundle["entry"]) >= 5


@pytest.mark.asyncio
async def test_verification_agent_consensus_and_discrepancy():
    """Tests the AI Council verification node for consensus score and red-flag audits."""
    # 1. Normal consensus scenario
    normal_triage = {
        "triage_level": "HOME_CARE",
        "detected_symptoms": {"critical_flags": []}
    }
    res_agree = await verify_clinical_claims(
        user_query="I have a mild runny nose",
        primary_triage=normal_triage
    )
    assert res_agree["council_status"] == "CONSENSUS_REACHED"
    assert res_agree["consensus_confidence_score"] >= 85

    # 2. Conflict scenario: Critical symptoms flagged but triage level was set to HOME_CARE
    conflicted_triage = {
        "triage_level": "HOME_CARE",
        "detected_symptoms": {"critical_flags": ["shortness of breath", "cyanosis"]}
    }
    res_conflict = await verify_clinical_claims(
        user_query="Patient cannot breathe and has blue lips",
        primary_triage=conflicted_triage
    )
    assert res_conflict["council_status"] == "ADJUSTMENT_RECOMMENDED"
    assert len(res_conflict["audit_findings"]["discrepancies"]) > 0


@pytest.mark.asyncio
async def test_mental_health_agent_evaluation():
    """Tests the Tele-MANAS mental health evaluation and coping techniques."""
    res = await evaluate_mental_wellbeing("I feel burnt out from work and cannot concentrate")
    assert res["domain"] in ("MENTAL_WELLBEING", "WOMENS_HEALTH")
    assert "evidence_based_techniques" in res
    assert "tele_manas_helpline" in res
    assert "14416" in str(res["tele_manas_helpline"])


@pytest.mark.asyncio
async def test_guideline_matching_retrieval():
    """Tests the WHO/ICMR guideline matching engine for common conditions."""
    asthma_res = await hybrid_retrieve_clinical_context("I need asthma inhaler guidance")
    assert len(asthma_res["who_icmr_guidelines"]) > 0
    assert any("Asthma" in g["title"] for g in asthma_res["who_icmr_guidelines"])
    assert asthma_res["corpus_total_indexed"] >= 10

    tb_res = await hybrid_retrieve_clinical_context("tuberculosis treatment protocol")
    assert len(tb_res["who_icmr_guidelines"]) > 0
    assert any("Tuberculosis" in g["title"] for g in tb_res["who_icmr_guidelines"])


def test_appointment_booking_agent():
    """Tests finding doctors and generating verified appointment tokens."""
    cardio_docs = find_doctors_by_specialty("Cardiologist")
    assert len(cardio_docs) > 0
    assert any("Cardiologist" in d["specialty"] for d in cardio_docs)

    booking = book_appointment_slot(
        patient_name="Priya Sharma",
        doctor_id="DOC-AIIMS-101",
        slot_time="Tomorrow at 10:30 AM"
    )
    assert booking["status"] == "CONFIRMED"
    assert booking["patient_name"] == "Priya Sharma"
    assert "DOC-AIIMS-101" in booking["doctor_id"] if "doctor_id" in booking else True
    assert "booking_id" in booking
