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
