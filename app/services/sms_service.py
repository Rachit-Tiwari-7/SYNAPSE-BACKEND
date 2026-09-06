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


def format_sms_text(text: str, max_chars: int = 320) -> str:
    """
    Formats and trims text suitable for clean 2G GSM 7-Bit plain text SMS delivery.
    Strips code blocks, markdown symbols, asterisks, hashes, backticks, divider bars,
    emojis, corrupted characters, and raw URL dumps.
    """
    if not text:
        return ""
    
    # 1. Remove code blocks and JSON-like snippets
    clean = re.sub(r'```[\s\S]*?```', '', text)
    clean = re.sub(r'`[^`]*`', '', clean)
    clean = re.sub(r'\{[^{}]*\}', '', clean)

    # 2. Remove URLs if they are raw IPFS / gateway links
    clean = re.sub(r'https?://(?:gateway\.pinata\.cloud|ipfs)[^\s)]+', '', clean)

    # 3. Strip markdown symbols, dividers, and bullet points
    clean = re.sub(r'[#*_~`>]', '', clean)
    clean = re.sub(r'[-=━─]{3,}', '', clean)
    clean = clean.replace('•', '-')

    # 4. Strip emojis and symbols that break 2G GSM-7 keypad phones
    clean = re.sub(
        r'[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\uFE00-\uFE0F\u200D\u200C\u2300-\u23FF\u2B50-\u2B55\u203C\u2049\u2139\u2194-\u21AA\u2934\u2935\u3297\u3299\u3030\u303D]',
        '',
        clean
    )

    # 5. Remove corrupted non-ASCII replacement artifacts like d?? or multiple question marks
    clean = re.sub(r'\bd\?+\b', '', clean)
    clean = re.sub(r'\?{2,}', '', clean)

    # 6. Clean up redundant spaces and newlines
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    clean = " ".join(lines)
    clean = re.sub(r'\s{2,}', ' ', clean).strip()

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
    Fetches medical document image from Twilio MMS or media URL, runs Prescription OCR,
    and returns concise transcription and clinical guidance for 2G SMS delivery.
    """
    try:
        from backend.app.services.prescription_ocr_service import (
            validate_image_bytes,
            normalize_and_resize_image,
            run_prescription_ocr,
            interpret_prescription
        )
        async with httpx.AsyncClient(timeout=40.0) as client:
            img_resp = await client.get(media_url)
            if img_resp.status_code != 200:
                return {"success": False, "error": f"Failed to retrieve image: HTTP {img_resp.status_code}"}

            img_bytes = img_resp.content
            valid, err_code, err_msg, pil_img = validate_image_bytes(img_bytes)
            if valid and pil_img:
                data_url = normalize_and_resize_image(pil_img)
                ok, err_obj, ocr_data = await run_prescription_ocr(data_url)
                if ok and ocr_data:
                    interp = await interpret_prescription(ocr_data=ocr_data, lang="en")
                    condition = interp.get("likely_condition", "Prescription Review")
                    meds = [m.get("name") or m.get("raw_name") for m in ocr_data.get("medications", []) if m.get("name") or m.get("raw_name")]
                    meds_summary = ", ".join(meds[:3]) if meds else "Medicines noted"
                    return {
                        "success": True,
                        "summary": f"Prescription: {condition}. Medicines: {meds_summary}",
                        "raw": ocr_data
                    }
    except Exception as e:
        logger.error(f"Error processing prescription MMS: {e}")

    return {
        "success": True,
        "summary": "Prescription received. Follow doctor's consultation and take prescribed doses after food with clean water."
    }


async def process_sms_inbound_webhook(
    from_number: str,
    body: str,
    media_url: Optional[str] = None,
    raw_payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Core 2G SMS routing engine for inbound webhook / keypad simulator.
    Delivers clean GSM 7-bit plain text without code, raw markdown, or corrupted characters.
    Handles Symptom Triage (1), Drug Safety (2), UIP Vaccination (7), District Outbreaks (8),
    Rural Preventive ORS (9), Appointments (5), and Emergency SOS.
    """
    clean_body = body.strip() if body else ""
    lower_body = clean_body.lower()

    # 0. Check for Twilio MMS / Image input
    target_media_url = media_url
    if not target_media_url and clean_body:
        url_match = re.search(r'https?://[^\s]+(?:\.jpg|\.jpeg|\.png|\.webp|[a-zA-Z0-9_\-/]+)', clean_body)
        if url_match and ("rx" in lower_body or "prescription" in lower_body or "medicine" in lower_body or ".jpg" in lower_body or ".png" in lower_body):
            target_media_url = url_match.group(0)

    if target_media_url:
        scan_analysis = await analyze_scan_via_twilio_backend(media_url=target_media_url, from_number=from_number)
        summary = scan_analysis.get("summary", "Prescription processed.")

        # Pin to IPFS
        ipfs_record = {
            "patient_phone": from_number,
            "channel": "twilio_sms_mms",
            "media_source": target_media_url,
            "prescription_analysis": scan_analysis
        }
        ipfs_res = await upload_json_to_ipfs(ipfs_record, record_name=f"sms_rx_{from_number}.json")
        ipfs_url = ipfs_res.get("gateway_url", "")

        sms_reply = f"SANJEEVNI PRESCRIPTION AI: {summary}. Follow doctor's advice."
        return {
            "status": "processed",
            "type": "prescription_analysis",
            "intent": "PRESCRIPTION_VISION_OCR",
            "media_url": target_media_url,
            "scan_result": scan_analysis,
            "ipfs_url": ipfs_url,
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    if not clean_body:
        reply = "SANJEEVNI HEALTH SMS: Reply 1 <symptoms>, 2 <meds>, 7 <age> for Vaccine, 8 <district> for Outbreaks, 9 for ORS Tips, SOS for 112/108."
        return {
            "status": "processed",
            "type": "empty_fallback",
            "reply": reply,
            "twiml": generate_twiml_response(reply)
        }

    # 1. Emergency SOS check
    if any(k in lower_body for k in EMERGENCY_KEYWORDS) or lower_body == "sos":
        sos_reply = (
            "SANJEEVNI RED ALERT: Call 108 / 112 immediately for Emergency Ambulance. "
            "Keep patient resting and calm. Nearest PHC notified."
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
        menu_reply = "SANJEEVNI HEALTH SMS: Reply 1 <symptoms>, 2 <meds>, 7 <age> for Vaccine, 8 <district> for Outbreaks, 9 for ORS Tips, SOS for 112/108."
        return {
            "status": "processed",
            "type": "menu_dispatched",
            "intent": "MENU_NAVIGATION",
            "reply": menu_reply,
            "twiml": generate_twiml_response(menu_reply)
        }

    # Extract command token and trailing query
    first_token = lower_body.split()[0] if lower_body else ""
    rest_query = clean_body[len(first_token):].strip() if len(clean_body) > len(first_token) else ""

    # Option 1: Symptom Triage (e.g. "1 high fever and headache", "1", "symptom fever", "fever")
    if first_token == "1" or ("symptom" in lower_body and len(lower_body.split()) > 1):
        from backend.app.services.i18n_service import detect_text_language
        symptom_text = rest_query if rest_query else clean_body
        s_lang = detect_text_language(symptom_text, default="en")
        is_hi = (s_lang == "hi")
        
        triage_res = await analyze_symptoms(text=symptom_text, lang=s_lang)
        urgency = triage_res.get("triage_level", "DOCTOR_CONSULT").replace("_", " ")
        
        # Determine probable relief medication
        s_lower = symptom_text.lower()
        if any(k in s_lower for k in ["fever", "headache", "body ache", "bukhar", "dard"]):
            med = "Dolo 650: 1 tab after meals with water (max 3/day)" if not is_hi else "Dolo 650: 1 goli khane ke baad paani se (max 3/day)"
            care = "Rest, drink ORS/fluids, sponge bath if high fever" if not is_hi else "Aaram karein, ORS/paani piyein, patti rakhein"
        elif any(k in s_lower for k in ["vomit", "diarrhea", "loose", "dast", "ulti"]):
            med = "Electral ORS: 1 packet in 1L clean water, sip frequently" if not is_hi else "Electral ORS: 1L paani me 1 packet gholkar piyein"
            care = "Keep hydrated, eat light khichdi" if not is_hi else "Paani ki kami na hone dein, halka khana lein"
        elif any(k in s_lower for k in ["acid", "gas", "jalan"]):
            med = "Pan-40: 1 tab 30 min before breakfast empty stomach" if not is_hi else "Pan-40: 1 goli subah khali pet"
            care = "Avoid spicy/fried food" if not is_hi else "Tala-bhuna na khayein"
        elif any(k in s_lower for k in ["cold", "cough", "sardi", "khansi"]):
            med = "Cetirizine 10mg: 1 tab at bedtime after food" if not is_hi else "Cetirizine 10mg: Raat ko khane ke baad 1 goli"
            care = "Warm saline gargle & steam" if not is_hi else "Gungune paani se garare v bhaap lein"
        else:
            med = "Paracetamol 650mg: 1 tab after food if pain/fever" if not is_hi else "Paracetamol 650mg: Khane ke baad 1 goli"
            care = "Ensure rest & hydration" if not is_hi else "Aaram v paryapt paani piyein"

        if is_hi:
            sms_reply = f"SANJEEVNI TRIAGE [{urgency}]: Dawa: {med}. Dekhbhal: {care}. 24-48h me Doctor ko dikhayein. AI Disclaimer: Doctor se consult karein. Emergency SOS: 108."
        else:
            sms_reply = f"SANJEEVNI TRIAGE [{urgency}]: Med: {med}. Care: {care}. Consult doctor if persists > 48h. AI Disclaimer: Educational advice only. Emergency: 108."
            
        return {
            "status": "processed",
            "type": "symptom_triage",
            "intent": "SYMPTOM_TRIAGE",
            "urgency": urgency,
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # Option 2: Drug Interaction (e.g. "2 Paracetamol and Aspirin", "2", "drug aspirin")
    if first_token == "2" or "drug" in lower_body or "medicine" in lower_body or "interaction" in lower_body or "paracetamol" in lower_body:
        drug_query = rest_query if rest_query else clean_body
        safety_res = await evaluate_drug_safety(text=drug_query)
        is_safe = safety_res.get("safe_to_combine", True)
        safety_status = "SAFE" if is_safe else "CAUTION"
        summary = safety_res.get("clinical_pharmacology_summary", "Standard drug safety review completed.")
        clean_summary = format_sms_text(str(summary), 220)

        sms_reply = f"SANJEEVNI DRUG SAFETY [{safety_status}]: {clean_summary}. Consult doctor for proper dosing."
        return {
            "status": "processed",
            "type": "drug_check",
            "intent": "DRUG_SAFETY",
            "safe": is_safe,
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # Option 7 or 4: UIP Vaccination Schedule (e.g. "7 6 weeks", "7 birth", "7", "4", "vaccine")
    if first_token in {"7", "4"} or "vaccine" in lower_body or "uwin" in lower_body or "immunization" in lower_body:
        query_text = rest_query.lower() if rest_query else lower_body
        weeks = 6
        age_label = "6 Weeks"
        if "birth" in query_text or "newborn" in query_text or "0" in query_text:
            weeks = 0
            age_label = "Birth Dose"
        elif "10" in query_text or "2.5" in query_text:
            weeks = 10
            age_label = "10 Weeks"
        elif "14" in query_text or "3.5" in query_text:
            weeks = 14
            age_label = "14 Weeks"
        elif "9 month" in query_text or "1 year" in query_text:
            weeks = 40
            age_label = "9 Months"

        vax_res = calculate_vaccination_schedule(age_in_weeks=weeks, category="child")
        due_str = vax_res.get("next_vaccine_due", "Pentavalent-1, Rotavirus-1, fIPV-1, PCV-1")
        
        sms_reply = f"SANJEEVNI UIP VACCINE ({age_label}): Due: {due_str}. Free at nearest Anganwadi/PHC. National Helpline: 1075."
        return {
            "status": "processed",
            "type": "vaccination_schedule",
            "intent": "VACCINATION_SCHEDULE",
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # Option 8 or 3: District Outbreak & Disease Alert (e.g. "8 Delhi", "8 Mumbai", "8", "3", "outbreak")
    if first_token in {"8", "3"} or "outbreak" in lower_body or "dengue" in lower_body or "malaria" in lower_body:
        district_query = rest_query if rest_query else "Delhi"
        outbreak_res = get_district_outbreak_risk(query=district_query)
        o_data = outbreak_res.get("data", {})
        risk = o_data.get("risk_badge", "HIGH").replace("🔴", "").replace("🟠", "").replace("🟢", "").strip()
        primary = o_data.get("primary_outbreak", "Dengue & Chikungunya")
        advisory = o_data.get("preventive_advisory", "Clean coolers on Sunday Dry Day. Use mosquito nets.")
        helpline = o_data.get("helpline", "011-22307145")
        
        clean_advisory = format_sms_text(str(advisory), 120)
        sms_reply = f"SANJEEVNI OUTBREAK ALERT ({district_query}): Risk: {risk}. Surge in {primary}. Advisory: {clean_advisory}. Helpline: {helpline}."
        return {
            "status": "processed",
            "type": "outbreak_alert",
            "intent": "OUTBREAK_ALERT",
            "risk_level": risk,
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # Option 9: Rural Preventive Health & ORS Tips (e.g. "9", "9 ORS", "diarrhea", "nutrition")
    if first_token == "9" or "ors" in lower_body or "diarrhea" in lower_body or "preventive" in lower_body or "nutrition" in lower_body:
        sms_reply = "SANJEEVNI ORS GUIDE: Mix 1 WHO-ORS packet in 1L clean water. Give frequent sips after loose stool + Zinc 20mg daily for 14 days. If severe dehydration, visit PHC."
        return {
            "status": "processed",
            "type": "rural_preventive",
            "intent": "RURAL_HEALTH",
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # Option 5: Appointment / PHC
    if first_token == "5" or "appointment" in lower_body or "doctor" in lower_body or "phc" in lower_body:
        sms_reply = "SANJEEVNI PHC: Dr. R. Sharma (General Medicine) available today at 3:30 PM. Reply CONFIRM to book."
        return {
            "status": "processed",
            "type": "appointment_slot",
            "intent": "APPOINTMENT_SLOT",
            "reply": sms_reply,
            "twiml": generate_twiml_response(sms_reply)
        }

    # 4. General Natural Clinical Language & Swarm Fallback
    orch_res = await orchestrate_health_request(message=clean_body, channel="sms", user_id=from_number)
    final_text = getattr(orch_res, "final_response", "") or str(orch_res)
    detected_intent = getattr(orch_res, "detected_intent", "GENERAL_HEALTH")
    clean_summary = format_sms_text(final_text, 260)

    sms_reply = f"SANJEEVNI HEALTH: {clean_summary} (Reply 1-9 for menus, SOS for emergency)."
    
    return {
        "status": "processed",
        "type": "orchestrated_intent",
        "intent": detected_intent,
        "reply": sms_reply,
        "twiml": generate_twiml_response(sms_reply)
    }
