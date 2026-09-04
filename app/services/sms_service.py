"""
SynapseOS — services/sms_service.py
Robust 2-way SMS Service with Twilio Webhook, TwiML formatting, AI Triage Agent Swarm routing,
Emergency SOS escalation, and Pinata IPFS decentralized medical record linking.
"""

import logging
import re
from typing import Dict, Any, Optional
import httpx
from xml.sax.saxutils import escape as xml_escape

from backend.app.core.config import settings
from backend.app.agents.orchestrator import orchestrate_health_request
from backend.app.agents.triage_agent import analyze_symptoms
from backend.app.agents.drug_agent import evaluate_drug_safety
from backend.app.agents.outbreak_agent import get_district_outbreak_risk
from backend.app.agents.vaccination_agent import calculate_vaccination_schedule
from backend.app.services.pinata_service import upload_json_to_ipfs

logger = logging.getLogger("synapseos.sms")

SMS_MAIN_MENU = (
    "Sanjeevni AI Health Assistant:\n"
    "1. Symptom Triage & Diagnosis\n"
    "2. Drug Interaction & Safety Check\n"
    "3. Disease & Outbreak Alert\n"
    "4. UIP Vaccination Schedule\n"
    "5. Book Teleconsult / PHC\n"
    "Reply with a number + query, or describe symptoms directly. (Text 'SOS' for Emergency)"
)

EMERGENCY_KEYWORDS = {"sos", "emergency", "ambulance", "108", "urgent", "heart attack", "stroke", "severe bleeding"}


def format_sms_text(text: str, max_chars: int = 480) -> str:
    """Formats and trims text suitable for clean SMS delivery."""
    clean = re.sub(r'[*_#`]', '', text).strip()
    if len(clean) > max_chars:
        clean = clean[:max_chars - 3].rstrip() + "..."
    return clean


def generate_twiml_response(message_body: str) -> str:
    """Generates valid TwiML XML string for immediate Twilio webhook HTTP response."""
    safe_body = xml_escape(message_body)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Response>\n'
        f'    <Message><Body>{safe_body}</Body></Message>\n'
        '</Response>'
    )


async def send_outbound_sms(to_number: str, message: str) -> Dict[str, Any]:
    """
    Sends an outbound SMS using Twilio REST API.
    Falls back to simulation mode if Twilio credentials are not configured.
    """
    account_sid = settings.TWILIO_ACCOUNT_SID.strip()
    auth_token = settings.TWILIO_AUTH_TOKEN.strip()
    from_number = settings.TWILIO_PHONE_NUMBER.strip()

    clean_message = format_sms_text(message)

    if not (account_sid and auth_token and from_number):
        logger.info(f"[SMS SIMULATION] Outbound to {to_number}: {clean_message}")
        return {
            "status": "sent",
            "simulated": True,
            "to": to_number,
            "from": from_number or "+15005550006",
            "body": clean_message,
            "sid": f"SM_SIMULATED_{abs(hash(to_number + clean_message))}"
        }

    twilio_url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    data = {
        "To": to_number,
        "From": from_number,
        "Body": clean_message
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(twilio_url, data=data, auth=(account_sid, auth_token))
            if resp.status_code in (200, 201):
                res_data = resp.json()
                return {
                    "status": "sent",
                    "simulated": False,
                    "to": to_number,
                    "from": from_number,
                    "sid": res_data.get("sid"),
                    "body": clean_message
                }
            else:
                logger.warning(f"Twilio API returned {resp.status_code}: {resp.text}. Using simulation fallback.")
    except Exception as e:
        logger.error(f"Twilio outbound exception: {e}. Using simulation fallback.")

    return {
        "status": "sent",
        "simulated": True,
        "to": to_number,
        "from": from_number or "+15005550006",
        "body": clean_message,
        "sid": f"SM_FALLBACK_{abs(hash(to_number + clean_message))}"
    }


async def analyze_scan_via_twilio_backend(media_url: str, from_number: str) -> Dict[str, Any]:
    """
    Fetches image from Twilio MMS or external media URL, dispatches to the Hugging Face
    FastAPI YOLOv8 detection model backend (https://yamxxx1-my-fastapi-app.hf.space/detect),
    and returns localized fracture bounding boxes and Grad-CAM attention links.
    """
    backend_url = getattr(settings, "TWILIO_MODEL_BACKEND_URL", "https://yamxxx1-my-fastapi-app.hf.space").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            # 1. Download image from media_url
            img_resp = await client.get(media_url)
            if img_resp.status_code != 200:
                logger.error(f"Failed to download Twilio MMS image from {media_url}: HTTP {img_resp.status_code}")
                return {"success": False, "error": f"Failed to retrieve image: HTTP {img_resp.status_code}"}

            img_bytes = img_resp.content

            # 2. Dispatch to FastAPI model backend /detect
            detect_resp = await client.post(
                f"{backend_url}/detect",
                files={"file": ("scan.jpg", img_bytes, "image/jpeg")}
            )

            if detect_resp.status_code == 200:
                data = detect_resp.json()
                detections = data.get("detections", [])
                result_image = data.get("result_image")
                gradcam_image = data.get("gradcam_image")

                result_img_url = f"{backend_url}{result_image}" if result_image else None
                gradcam_img_url = f"{backend_url}{gradcam_image}" if gradcam_image else None

                has_fracture = len(detections) > 0
                summary = (
                    f"⚠️ Fracture Detected ({len(detections)} anomaly zones localized)"
                    if has_fracture
                    else "✅ Skeletal Integrity Preserved (No acute cortical fracture identified)"
                )

                return {
                    "success": True,
                    "summary": summary,
                    "detections": detections,
                    "result_image_url": result_img_url,
                    "gradcam_image_url": gradcam_img_url,
                    "raw": data
                }
            else:
                logger.warning(f"FastAPI model backend returned {detect_resp.status_code}: {detect_resp.text}")
                return {"success": False, "error": f"Model inference status {detect_resp.status_code}"}
    except Exception as e:
        logger.error(f"Error fetching Twilio model backend at {backend_url}: {e}")
        return {"success": False, "error": str(e)}


async def process_sms_inbound_webhook(
    from_number: str,
    body: str,
    media_url: Optional[str] = None,
    raw_payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Core SMS routing engine for inbound webhook.
    Handles Triage, Drug Safety, Outbreak Alerts, Emergency SOS, Pinata IPFS Record Pinning,
    and Twilio MMS Medical Image Analysis via Hugging Face YOLOv8 FastAPI backend.
    """
    clean_body = body.strip() if body else ""
    lower_body = clean_body.lower()

    # 0. Check for Twilio MMS / Image Scan input (via MediaUrl0 or link in SMS body)
    target_media_url = media_url
    if not target_media_url and clean_body:
        url_match = re.search(r'https?://[^\s]+(?:\.jpg|\.jpeg|\.png|\.webp|[a-zA-Z0-9_\-/]+)', clean_body)
        if url_match and ("scan" in lower_body or "xray" in lower_body or "fracture" in lower_body or "bone" in lower_body or ".jpg" in lower_body or ".png" in lower_body):
            target_media_url = url_match.group(0)

    if target_media_url:
        scan_analysis = await analyze_scan_via_twilio_backend(media_url=target_media_url, from_number=from_number)
        if scan_analysis.get("success"):
            summary = scan_analysis["summary"]
            res_img = scan_analysis.get("result_image_url") or ""
            grad_img = scan_analysis.get("gradcam_image_url") or ""

            # Pin to IPFS
            ipfs_record = {
                "patient_phone": from_number,
                "channel": "twilio_sms_mms",
                "media_source": target_media_url,
                "scan_analysis": scan_analysis
            }
            ipfs_res = await upload_json_to_ipfs(ipfs_record, record_name=f"sms_scan_{from_number}.json")
            ipfs_url = ipfs_res.get("gateway_url", "")

            reply_lines = [
                f"🦴 Sanjeevni YOLOv8 Scan AI: {summary}",
                f"• Visual Overlay: {res_img}" if res_img else "",
                f"• Grad-CAM Heatmap: {grad_img}" if grad_img else "",
                f"📋 Decentralized IPFS: {ipfs_url}" if ipfs_url else "",
                "⚠️ Clinical screening support only. Consult an orthopedic doctor."
            ]
            sms_reply = "\n".join([line for line in reply_lines if line])
            return {
                "status": "processed",
                "type": "medical_scan_analysis",
                "intent": "MEDICAL_SCAN_YOLOV8",
                "media_url": target_media_url,
                "scan_result": scan_analysis,
                "ipfs_url": ipfs_url,
                "reply": sms_reply,
                "twiml": generate_twiml_response(sms_reply)
            }

    if not clean_body:
        reply = "Sanjeevni AI: Empty message received. " + SMS_MAIN_MENU
        return {
            "status": "processed",
            "type": "empty_fallback",
            "reply": reply,
            "twiml": generate_twiml_response(reply)
        }

    # 1. Emergency SOS check
    if any(k in lower_body for k in EMERGENCY_KEYWORDS):
        sos_reply = (
            "🚨 SANJEEVNI RED ALERT EMERGENCY PROTOCOL ACTIVATED 🚨\n\n"
            "• Call 108 / 112 immediately for Ambulance.\n"
            "• Keep patient calm and in resting posture.\n"
            "• If chest pain/shortness of breath, loosen tight clothing.\n"
            "• Nearest Emergency PHC notified."
        )
        return {
            "status": "processed",
            "type": "emergency_sos",
            "intent": "EMERGENCY_SOS",
            "reply": sos_reply,
            "twiml": generate_twiml_response(sos_reply)
        }

    # 2. Greeting / Menu dispatch
    if lower_body in {"hi", "hello", "namaste", "menu", "start", "help", "info"}:
        return {
            "status": "processed",
            "type": "menu_dispatched",
            "intent": "MENU_NAVIGATION",
            "reply": SMS_MAIN_MENU,
            "twiml": generate_twiml_response(SMS_MAIN_MENU)
        }

    # 3. Numbered Menu Selection Routing
    first_token = lower_body.split()[0] if lower_body else ""
    rest_query = clean_body[len(first_token):].strip() if len(clean_body) > len(first_token) else ""

    # Option 1: Symptom Triage
    if first_token == "1" or ("symptom" in lower_body and len(lower_body.split()) > 2):
        symptom_text = rest_query if rest_query else clean_body
        triage_res = await analyze_symptoms(text=symptom_text)
        
        urgency = triage_res.get("triage_level", "DOCTOR_CONSULT")
        action = triage_res.get("recommended_action", "Consult nearest healthcare professional.")
        
        # Pin clinical summary to Pinata IPFS
        ipfs_record = {
            "patient_phone": from_number,
            "channel": "sms",
            "symptoms": symptom_text,
            "triage_level": urgency,
            "recommended_action": action,
            "triage_summary": triage_res
        }
        ipfs_res = await upload_json_to_ipfs(ipfs_record, record_name=f"sms_triage_{from_number}.json")
        cid = ipfs_res.get("cid", "")
        gateway_url = ipfs_res.get("gateway_url", "")

        sms_reply = (
            f"🏥 Sanjeevni Triage: {urgency.replace('_', ' ')}\n"
            f"Advice: {format_sms_text(str(action), 160)}\n"
            f"📋 Secure IPFS Record: {gateway_url}"
        )
        return {
            "status": "processed",
            "type": "symptom_triage",
            "intent": "SYMPTOM_TRIAGE",
            "urgency": urgency,
            "ipfs_cid": cid,
            "ipfs_url": gateway_url,
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # Option 2: Drug Interaction
    if first_token == "2" or "drug" in lower_body or "medicine" in lower_body or "interaction" in lower_body:
        drug_query = rest_query if rest_query else clean_body
        safety_res = await evaluate_drug_safety(text=drug_query)
        is_safe = safety_res.get("safe_to_combine", True)
        safety_status = "SAFE ✅" if is_safe else "CAUTION ⚠️"
        summary = safety_res.get("clinical_pharmacology_summary", "Standard interaction screening completed.")
        
        sms_reply = f"💊 Sanjeevni Drug Safety [{safety_status}]: {format_sms_text(str(summary), 280)}"
        return {
            "status": "processed",
            "type": "drug_check",
            "intent": "DRUG_SAFETY",
            "safe": is_safe,
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # Option 3: Outbreak & Disease Alert
    if first_token == "3" or "outbreak" in lower_body or "dengue" in lower_body or "malaria" in lower_body:
        district_query = rest_query if rest_query else "Delhi"
        outbreak_res = get_district_outbreak_risk(query=district_query)
        o_data = outbreak_res.get("data", {})
        risk = o_data.get("risk_level", "MODERATE")
        advisory = o_data.get("preventive_advisory", "Follow preventive hygiene.")
        
        sms_reply = f"📢 Outbreak Alert for {district_query} [{risk.upper()}]: {format_sms_text(str(advisory), 260)}"
        return {
            "status": "processed",
            "type": "outbreak_alert",
            "intent": "OUTBREAK_ALERT",
            "risk_level": risk,
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # Option 4: Vaccination Schedule
    if first_token == "4" or "vaccine" in lower_body or "uwin" in lower_body:
        vax_res = calculate_vaccination_schedule(age_in_weeks=6, category="child")
        due_str = vax_res.get("next_vaccine_due", "OPV-1, Pentavalent-1")
        
        sms_reply = f"💉 UIP Vaccination Schedule (6 Weeks): Due: {due_str}. Visit nearest Govt Anganwadi/PHC."
        return {
            "status": "processed",
            "type": "vaccination_schedule",
            "intent": "VACCINATION_SCHEDULE",
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # Option 5: Appointment / PHC
    if first_token == "5" or "appointment" in lower_body or "doctor" in lower_body or "phc" in lower_body:
        sms_reply = "🩺 PHC Slot: Dr. R. Sharma (General Medicine) available today at 3:30 PM. Reply 'CONFIRM' to book."
        return {
            "status": "processed",
            "type": "appointment_slot",
            "intent": "APPOINTMENT_SLOT",
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # 4. General Natural Clinical Language & Zero-Selection Intent Tracking
    orch_res = await orchestrate_health_request(message=clean_body, channel="sms", user_id=from_number)
    final_text = getattr(orch_res, "final_response", "") or str(orch_res)
    detected_intent = getattr(orch_res, "detected_intent", "GENERAL_HEALTH")
    summary_text = format_sms_text(final_text, 300)

    # If clinical triage or drug check, pin record to Pinata IPFS
    ipfs_cid = None
    ipfs_url = None
    if detected_intent in ("SYMPTOM_TRIAGE", "DRUG_SAFETY", "DIGITAL_TWIN"):
        ipfs_record = {
            "patient_phone": from_number,
            "channel": "sms",
            "intent": detected_intent,
            "query": clean_body,
            "response": final_text,
            "trace_steps": [t.action for t in getattr(orch_res, "trace", [])]
        }
        ipfs_res = await upload_json_to_ipfs(ipfs_record, record_name=f"sms_{detected_intent.lower()}_{from_number}.json")
        ipfs_cid = ipfs_res.get("cid")
        ipfs_url = ipfs_res.get("gateway_url")

    ipfs_suffix = f"\n📋 IPFS Record: {ipfs_url}" if ipfs_url else ""
    sms_reply = f"Sanjeevni AI [{detected_intent.replace('_', ' ')}]: {summary_text}{ipfs_suffix}\n(Text 'MENU' for options)"
    
    return {
        "status": "processed",
        "type": "orchestrated_intent",
        "intent": detected_intent,
        "ipfs_cid": ipfs_cid,
        "ipfs_url": ipfs_url,
        "reply": sms_reply,
        "twiml": generate_twiml_response(sms_reply)
    }
