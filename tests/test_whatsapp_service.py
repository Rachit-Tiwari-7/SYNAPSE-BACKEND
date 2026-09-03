"""
SynapseOS — tests/test_whatsapp_service.py
Unit tests for Meta Official WhatsApp Cloud API Webhook Ingestion, Multilingual Language Onboarding,
Interactive Menu Engine, Agent Swarm Routing, Medical Image Ingestion, and Emergency Protocols.
"""

import pytest
from backend.app.services.whatsapp_service import (
    process_whatsapp_inbound_webhook,
    MAIN_MENU_TEXT,
    format_response_for_whatsapp,
    format_compact_whatsapp_card,
    trigger_emergency_sos_whatsapp
)
from backend.app.core.session_manager import session_manager


@pytest.mark.asyncio
async def test_whatsapp_language_onboarding_and_selection():
    """Tests that a first-time greeting triggers language selection, and selecting '2' switches to Hindi."""
    test_phone = "919999988888"
    session_manager.reset_flow(test_phone)
    if test_phone in session_manager._sessions:
        session_manager._sessions[test_phone]["context"] = {}

    # 1. Greeting -> Prompt Language Selection
    greeting_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "100000000000000",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "15550234567", "phone_number_id": "100000000000000"},
                    "contacts": [{"profile": {"name": "Test User"}, "wa_id": test_phone}],
                    "messages": [{
                        "from": test_phone,
                        "id": "wamid.TEST_GREETING",
                        "timestamp": "1772185000",
                        "type": "text",
                        "text": {"body": "Hi"}
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    result = await process_whatsapp_inbound_webhook(greeting_payload)
    assert result["status"] == "processed"
    assert result["type"] == "menu_dispatched"
    assert result["sender"] == test_phone

    # 2. Select Language '2' (Hindi)
    lang_payload = {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "100000000000000",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "15550234567", "phone_number_id": "100000000000000"},
                    "contacts": [{"profile": {"name": "Test User"}, "wa_id": test_phone}],
                    "messages": [{
                        "from": test_phone,
                        "id": "wamid.TEST_LANG_SEL",
                        "timestamp": "1772185001",
                        "type": "text",
                        "text": {"body": "2"}
                    }]
                },
                "field": "messages"
            }]
        }]
    }
    result_lang = await process_whatsapp_inbound_webhook(lang_payload)
    assert result_lang["status"] == "processed"
    assert result_lang["type"] == "language_selected"
    assert result_lang["language"] == "hi"


@pytest.mark.asyncio
async def test_whatsapp_numbered_drug_check():
    """Tests Option 2 for drug safety check via WhatsApp."""
    payload = {
        "event": "onMessage",
        "data": {
            "from": "919876543210",
            "body": "2 Can I take Aspirin with Warfarin?",
            "type": "text"
        }
    }
    result = await process_whatsapp_inbound_webhook(payload)
    assert result["status"] == "processed"
    assert result["type"] == "drug_check"
    assert result["dispatch"]["delivered"] is True


@pytest.mark.asyncio
async def test_whatsapp_emergency_sos():
    """Tests triggering emergency SOS keyword over WhatsApp."""
    payload = {
        "event": "onMessage",
        "data": {
            "from": "919876543210",
            "body": "SOS",
            "type": "text"
        }
    }
    result = await process_whatsapp_inbound_webhook(payload)
    assert result["status"] == "processed"
    assert result["type"] == "emergency_sos"


@pytest.mark.asyncio
async def test_whatsapp_doctor_lookup():
    """Tests Option 5 for PM-JAY doctor search."""
    payload = {
        "event": "onMessage",
        "data": {
            "from": "919876543210",
            "body": "5 Cardiologist",
            "type": "text"
        }
    }
    result = await process_whatsapp_inbound_webhook(payload)
    assert result["status"] == "processed"
    assert result["type"] == "doctor_lookup"


@pytest.mark.asyncio
async def test_whatsapp_abha_schemes():
    """Tests Option 6 for ABHA and Ayushman Bharat scheme details."""
    payload = {
        "event": "onMessage",
        "data": {
            "from": "919876543210",
            "body": "6",
            "type": "text"
        }
    }
    result = await process_whatsapp_inbound_webhook(payload)
    assert result["status"] == "processed"
    assert result["type"] == "abha_info"


@pytest.mark.asyncio
async def test_whatsapp_image_scan_analysis():
    """Tests medical imaging upload over WhatsApp."""
    payload = {
        "event": "onMessage",
        "data": {
            "from": "919876543210",
            "body": "chest_scan.jpg",
            "caption": "Doctor, please check this chest scan for cough",
            "type": "image"
        }
    }
    result = await process_whatsapp_inbound_webhook(payload)
    assert result["status"] == "processed"
    assert result["type"] == "medical_image"
    assert "scan_summary" in result


@pytest.mark.asyncio
async def test_whatsapp_natural_symptom_triage():
    """Tests natural language clinical triage message."""
    payload = {
        "event": "onMessage",
        "data": {
            "from": "919876543210",
            "body": "I have mild sore throat, sneezing and runny nose",
            "type": "text"
        }
    }
    result = await process_whatsapp_inbound_webhook(payload)
    assert result["status"] == "processed"
    assert result["sender"] == "919876543210"
    assert result["reply_dispatched"]["delivered"] is True


@pytest.mark.asyncio
async def test_emergency_sos_direct_dispatch():
    """Tests direct 1-click SOS broadcast function."""
    res = await trigger_emergency_sos_whatsapp(
        emergency_contact="+919876543210",
        patient_name="Siddharth Sharma",
        location_coords="28.6139,77.2090",
        blood_group="B+",
        critical_symptoms="Severe breathlessness"
    )
    assert res["emergency_alert_dispatched"] is True
    assert res["contact_notified"] == "+919876543210"


def test_compact_whatsapp_card_emergency_triage_clean_diagnosis_and_indian_meds():
    """
    Verifies that the emergency triage short message:
    1. Extracts real diagnosis (e.g. Meningitis/Encephalitis) without leaking '### 🩺 Triage & Outbreak Assessment'.
    2. Includes Indian medication guidance with clear safety protocol against self-medication.
    3. Stays compact and contains quick shortcuts (5, sos, full).
    """
    sample_emergency_audit = """*🚨 Critical Clinical Summary: Immediate Emergency Action Required

**Patient Status:** 🔴 **EMERGENCY CARE (IMMEDIATE)**
**AI Council Confidence:** 98% Consensus | **Safety Protocol:** Strict Adherence to CNS Infection Triage Guidelines

*1. Executive Summary
Your reported symptoms—**fever, headache, and continuous vomiting**—constitute a **medical emergency**. This specific combination is a critical red flag for potentially life-threatening conditions, including **meningitis**, **encephalitis**, or severe dehydration with electrolyte imbalance.

**Do not wait. Do not drive yourself.** You require immediate professional medical evaluation to rule out central nervous system infections and metabolic crises.

*2. Immediate Action Plan
*   **Seek Emergency Care Now:** Call emergency services (e.g., 911/112) or have a companion take you to the nearest Emergency Department immediately.
*   **Do Not Self-Medicate:** **Strictly avoid** taking anti-emetics (anti-vomiting drugs), pain relievers (like paracetamol/ibuprofen), or fever reducers until evaluated by a doctor. These medications can mask critical neurological signs (such as neck stiffness) and delay accurate diagnosis.
*   **Hydration Caution:** If you are unable to keep fluids down, you are at high risk for rapid dehydration.
*   **Neck Stiffness Check:** While waiting for help, gently try to touch your chin to your chest.

*3. Clinical Rationale & Specialist Findings

• *🩺 Triage & Outbreak Assessment
*   **Critical Flags Detected:** Continuous vomiting, Fever, Headache.
*   **Risk Profile:** High risk for **Meningitis** or **Encephalitis**.

• *💊 Drug Safety & Pharmacology
*   **Current Medications:** None reported.
*   **Safety Note:** The AI Council confirms that withholding symptomatic relief is a **critical safety measure**.

• *🧠 AI Council Verification
*   **Status:** **CONSENSUS REACHED** (98% Confidence).
*   **Verdict:** The council unanimously supports the emergency triage.
"""
    card = format_compact_whatsapp_card(sample_emergency_audit)

    # 1. Verify No markdown heading leak and NO markdown syntax
    assert "###" not in card
    assert "*" not in card
    assert "_" not in card
    assert "🔴 SYNAPSE-OS EMERGENCY TRIAGE — CRITICAL" in card

    # 2. Verify Diagnosis extraction (Normal plain text)
    assert "🩺 Suspected Diagnosis:" in card
    assert any(term in card for term in ["Meningitis", "Encephalitis"])

    # 3. Verify Council consensus percentage
    assert "98%" in card

    # 4. Verify Indian Medication guidance with safety protocol
    assert "💊 Medications & Relief (India):" in card
    assert "Withhold self-medication" in card

    # 5. Verify Quick Shortcuts in normal text
    assert "Reply 5" in card
    assert "Reply sos" in card
    assert "Reply full" in card

    # 6. Verify Compact length (does not overflow chat)
    assert len(card) < 1150


def test_compact_whatsapp_card_homecare_indian_meds_and_instructions():
    """
    Verifies that for home care / mild illness, the short card includes:
    1. Clean normal text without markdown syntax (*, _, #).
    2. Real diagnosis.
    3. Indian OTC medicines (Dolo 650, Electral ORS) with specific eating/dosing instructions.
    4. Compact length.
    """
    sample_homecare_audit = """**🌿 Clinical Assessment & Care Guidance**

**Patient Status:** 🟢 **HOME CARE & MONITORING**
**AI Council Confidence:** 96% Agreement | **Safety Protocol:** Standard Upper Respiratory & Viral Care

*1. Executive Summary & Suspected Diagnosis
Your reported symptoms of runny nose, mild sore throat, and low-grade fever are consistent with **Acute Upper Respiratory Viral Infection (Common Cold)**. No emergency red flags detected.

*2. Immediate Action Plan
*   **Active Hydration:** Drink at least 2.5–3 liters of warm water, clear broths, and electrolyte fluids daily.
*   **Adequate Rest:** Prioritize 8+ hours of sleep and avoid strenuous physical exertion.

*3. Recommended Medications & Relief (India)
*   **Dolo 650 / Calpol (Paracetamol 650mg):** 1 tablet after meals (with water) every 6–8 hours as needed for fever or headache. Max 3 tablets/24h.
*   **Electral ORS:** 1 sachet dissolved in 1 liter clean water; sip throughout the day to sustain hydration.
*   **Cetzine / Okacet (Cetirizine 10mg):** 1 tablet at night after dinner if runny nose or sneezing disturbs sleep.
"""
    card = format_compact_whatsapp_card(sample_homecare_audit)

    # 1. Verify Status & Diagnosis in clean normal text (no markdown)
    assert "*" not in card
    assert "_" not in card
    assert "🟢 SYNAPSE-OS HOME CARE & MONITORING" in card
    assert "🩺 Suspected Diagnosis: Acute Upper Respiratory Viral Infection" in card

    # 2. Verify Indian Medications & Administration (how to eat/take)
    assert "💊 Medications & Relief (India):" in card
    assert "Dolo 650" in card
    assert "after meals" in card or "after food" in card
    assert "Electral ORS" in card

    # 3. Verify Shortcuts and length
    assert "Reply 5" in card
    assert "Reply full" in card
    assert len(card) < 1150


def test_compact_whatsapp_card_from_plain_unformatted_input():
    """
    Verifies that format_compact_whatsapp_card gracefully parses normal plain text
    without any markdown formatting and outputs clean normal text.
    """
    plain_input = """SYNAPSE-OS EMERGENCY CLINICAL ASSESSMENT
Patient Status: EMERGENCY CARE (IMMEDIATE)
Council Confidence: 99% Consensus

Executive Summary:
The patient has acute fever, severe persistent headache, and recurrent vomiting, representing a critical presentation.

Risk Profile: High risk for Meningitis or Acute Neurological Infection.

Immediate Action Plan:
1. Seek Emergency Care Now: Transport patient immediately to hospital emergency.
2. Do not self-medicate: Strictly avoid oral painkillers or anti-emetics before physician assessment.
"""
    card = format_compact_whatsapp_card(plain_input)
    assert "*" not in card
    assert "_" not in card
    assert "#" not in card
    assert "🔴 SYNAPSE-OS EMERGENCY TRIAGE — CRITICAL" in card
    assert "🩺 Suspected Diagnosis:" in card
    assert any(term in card for term in ["Meningitis", "Neurological"])
    assert "💊 Medications & Relief (India):" in card
    assert "Withhold self-medication" in card
    assert "Reply 5" in card
    assert len(card) < 1150


def test_informational_medicine_question_returns_concise_direct_answer_without_triage_slop():
    """
    Verifies that questions like 'what is calpol medicine for' return a direct,
    short, simple answer without the emergency triage badge, shortcuts, or fake diagnosis slop.
    """
    sample_medicine_response = """Calpol is a brand of **Paracetamol (Acetaminophen)**, used to reduce fever and relieve mild to moderate pain (such as headaches, toothaches, or body aches).

**How to take:**
• **Adults:** 500mg–1000mg after food with water every 4–6 hours (max 3000mg/day).
• **Children:** Dose depends on body weight and age; use pediatric drops/syrup as prescribed.
• **Safety:** Do not combine with other paracetamol-containing medicines (like Dolo or Crocin).

Seek immediate medical attention if you experience rash, breathing difficulty, or fever lasting > 3 days.
"""
    output = format_response_for_whatsapp(sample_medicine_response, compact=True)

    # Must be clean normal text (no markdown)
    assert "*" not in output
    assert "_" not in output
    assert "#" not in output

    # Must directly answer what Calpol is
    assert "Calpol is a brand of Paracetamol" in output
    assert "reduce fever" in output or "mild to moderate pain" in output
    assert "How to take:" in output

    # MUST NOT contain robotic triage template slop
    assert "SYNAPSE-OS HOME CARE & MONITORING" not in output
    assert "SYNAPSE-OS CLINICAL ASSESSMENT" not in output
    assert "Suspected Diagnosis:" not in output
    assert "The primary clinical impression is that this is a general informational inquiry" not in output
    assert "Reply 5" not in output
    assert "Reply sos" not in output

    # Must include clean footer
    assert "🌿 Powered by Synapse-OS Multi-Agent Swarm" in output



