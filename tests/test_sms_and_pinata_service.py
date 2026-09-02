"""
SynapseOS — tests/test_sms_and_pinata_service.py
Unit and Integration tests for SMS Webhook Ingestion, Twilio TwiML formatting,
AI Triage Swarm Routing, Emergency SOS, Outbound SMS, and Pinata IPFS storage.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app
from backend.app.services.sms_service import (
    process_sms_inbound_webhook,
    generate_twiml_response,
    send_outbound_sms,
    format_sms_text,
    SMS_MAIN_MENU
)
from backend.app.services.pinata_service import (
    upload_json_to_ipfs,
    upload_file_to_ipfs,
    get_ipfs_gateway_url,
    get_simulated_record
)


@pytest.mark.asyncio
async def test_sms_menu_and_greeting():
    """Tests that sending 'hi' or 'menu' via SMS returns the main health menu."""
    res = await process_sms_inbound_webhook(from_number="+919876543210", body="hi")
    assert res["status"] == "processed"
    assert res["type"] == "menu_dispatched"
    assert "Sanjeevni AI Health Assistant" in res["reply"]
    assert "<Response>" in res["twiml"]
    assert "<Body>" in res["twiml"]


@pytest.mark.asyncio
async def test_sms_emergency_sos():
    """Tests that sending SOS triggers emergency red alert protocol."""
    res = await process_sms_inbound_webhook(from_number="+919876543210", body="SOS severe chest pain")
    assert res["status"] == "processed"
    assert res["type"] == "emergency_sos"
    assert "108" in res["reply"]
    assert "RED ALERT" in res["reply"]
    assert "<Response>" in res["twiml"]


@pytest.mark.asyncio
async def test_sms_symptom_triage_with_pinata_ipfs():
    """Tests that symptom triage generates AI advice and pins record to Pinata IPFS."""
    res = await process_sms_inbound_webhook(
        from_number="+919876543210",
        body="1 High fever 102F and persistent dry cough for 3 days"
    )
    assert res["status"] == "processed"
    assert res["type"] == "symptom_triage"
    assert "Triage:" in res["reply"]
    assert res.get("ipfs_cid", "").startswith("Qm")
    assert "gateway.pinata.cloud/ipfs/Qm" in res["reply"]
    assert "<Response>" in res["twiml"]
    assert "<Body>" in res["twiml"]


@pytest.mark.asyncio
async def test_sms_drug_safety_check():
    """Tests option 2 for drug interaction over SMS."""
    res = await process_sms_inbound_webhook(
        from_number="+919876543210",
        body="2 Can I take Aspirin with Warfarin?"
    )
    assert res["status"] == "processed"
    assert res["type"] == "drug_check"
    assert "Drug Safety" in res["reply"]


@pytest.mark.asyncio
async def test_sms_outbreak_and_vaccination_routes():
    """Tests option 3 and option 4 over SMS."""
    res_outbreak = await process_sms_inbound_webhook(from_number="+919876543210", body="3 Delhi")
    assert res_outbreak["type"] == "outbreak_alert"

    res_vax = await process_sms_inbound_webhook(from_number="+919876543210", body="4")
    assert res_vax["type"] == "vaccination_schedule"
    assert "UIP Vaccination" in res_vax["reply"]


@pytest.mark.asyncio
async def test_pinata_json_and_file_upload_simulation():
    """Tests Pinata IPFS JSON and binary file upload with simulated storage."""
    # 1. Upload JSON
    test_json = {"patient_id": "TEST-PAT-01", "diagnosis": "Viral Fever", "confidence": 0.94}
    res_json = await upload_json_to_ipfs(test_json, record_name="test_record.json")
    assert res_json["status"] in ("pinned", "pinned_fallback")
    cid = res_json["cid"]
    assert cid.startswith("Qm")
    assert f"https://gateway.pinata.cloud/ipfs/{cid}" == res_json["gateway_url"]

    # Verify simulated record stored
    stored = get_simulated_record(cid)
    assert stored is not None
    assert stored["data"]["patient_id"] == "TEST-PAT-01"

    # 2. Upload File Bytes (PDF simulation)
    dummy_pdf_bytes = b"%PDF-1.4 Mock PDF clinical content for Sanjeevni"
    res_file = await upload_file_to_ipfs(dummy_pdf_bytes, filename="report.pdf")
    assert res_file["status"] in ("pinned", "pinned_fallback")
    assert res_file["cid"].startswith("Qm")


@pytest.mark.asyncio
async def test_outbound_sms_dispatch():
    """Tests outbound SMS dispatch with fallback/simulation handling."""
    res = await send_outbound_sms(
        to_number="+919876543210",
        message="Your Sanjeevni test results are ready."
    )
    assert res["status"] == "sent"
    assert res["to"] == "+919876543210"
    assert "SM_" in res["sid"]


@pytest.mark.asyncio
async def test_api_twilio_webhook_endpoint():
    """Integration test for POST /api/sms/webhook with standard Form data."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Inbound Twilio Webhook Form Data
        resp = await client.post(
            "/api/sms/webhook",
            data={
                "From": "+919876543210",
                "Body": "SOS urgent medical assistance",
                "To": "+15551234567"
            }
        )
        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]
        assert "<Response>" in resp.text
        assert "<Body>" in resp.text
        assert "EMERGENCY PROTOCOL" in resp.text


@pytest.mark.asyncio
async def test_api_sms_simulate_and_send_endpoints():
    """Integration test for POST /api/sms/simulate and POST /api/sms/send."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Simulate SMS
        sim_resp = await client.post(
            "/api/sms/simulate",
            json={"sender": "+919876543210", "message": "1 Fever and sore throat"}
        )
        assert sim_resp.status_code == 200
        sim_data = sim_resp.json()
        assert sim_data["status"] == "DELIVERED"
        assert sim_data["type"] == "symptom_triage"
        assert "ipfs_cid" in sim_data

        # Send Outbound SMS
        send_resp = await client.post(
            "/api/sms/send",
            json={"to_number": "+919876543210", "message": "Appointment Reminder"}
        )
        assert send_resp.status_code == 200
        assert send_resp.json()["status"] == "sent"


@pytest.mark.asyncio
async def test_sms_zero_mode_natural_intent_tracking():
    """Tests that sending unformatted free-text without selecting any mode automatically detects intent and responds."""
    # 1. Natural Language Drug Interaction query without number
    res_drug = await process_sms_inbound_webhook(
        from_number="+919876543210",
        body="Can I take paracetamol and ibuprofen together for a headache?"
    )
    assert res_drug["status"] == "processed"
    assert res_drug["intent"] in ("DRUG_SAFETY", "SYMPTOM_TRIAGE")
    assert "<Response>" in res_drug["twiml"]
    assert "Sanjeevni" in res_drug["reply"]

    # 2. Natural Language Vaccine query without number
    res_vax = await process_sms_inbound_webhook(
        from_number="+919876543210",
        body="What vaccines does a child need at 6 weeks under UIP?"
    )
    assert res_vax["status"] == "processed"
    assert res_vax["intent"] == "VACCINATION_SCHEDULE"
    assert "UIP Vaccination" in res_vax["reply"]

    # 3. Natural Language Free-form Clinical Query without number
    res_triage = await process_sms_inbound_webhook(
        from_number="+919876543210",
        body="I have had persistent severe stomach pain and dizziness since yesterday"
    )
    assert res_triage["status"] == "processed"
    assert res_triage["intent"] == "SYMPTOM_TRIAGE"
    assert "<Response>" in res_triage["twiml"]
    assert "Sanjeevni" in res_triage["reply"]
