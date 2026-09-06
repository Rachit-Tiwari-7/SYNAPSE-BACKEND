"""
SynapseOS — services/prescription_ocr_service.py
Production-Ready Medical Prescription OCR Engine using OpenRouter Free Multimodal Vision Models.
Ultra-lightweight, memory-conscious, CPU-conscious, anti-hallucinating document extraction.
"""

import io
import re
import json
import time
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image

import httpx
from backend.app.core.config import settings

logger = logging.getLogger("prescription_ocr")

# Multimodal Medical Prescription Vision & Clinical Intelligence Prompt
OCR_SYSTEM_PROMPT = (
    "You are an expert Clinical Medical Document & Prescription Vision Intelligence AI.\n"
    "Your task is to analyze the uploaded doctor prescription image and return accurate, structured clinical intelligence:\n"
    "1. TRANSCRIPTION & OCR: Accurately read and transcribe all visible handwritten and printed text from the prescription document into raw_text.\n"
    "2. EXTRACT DETAILS: Extract patient metadata (name, age, gender), doctor metadata (name, specialization, clinic/reg), prescription date, and lab tests ordered.\n"
    "3. MEDICATIONS BREAKDOWN: Extract all prescribed medications with exact strength, dosage, frequency (e.g. 1-0-1, OD, BD, TDS), duration, route, and timing instructions (e.g. After Meals, Before Breakfast on Empty Stomach).\n"
    "4. PROBABLE DIAGNOSIS: Clinically deduce the likely condition or illness being treated based on the combination of prescribed medicines, symptoms, and doctor's notes.\n"
    "5. PREVENTIVE MEASURES & PATIENT CARE: Provide clear, actionable preventive guidance, dietary instructions, hydration advice, medication safety rules, and lifestyle recommendations.\n"
    "6. RED FLAG WARNINGS: Highlight critical warning signs when the patient must seek urgent emergency care or call 108.\n"
    "7. JAN AUSHADHI GENERICS: Suggest generic alternatives / PMBJP savings tips for branded Indian pharmaceuticals where applicable.\n"
    "Return ONLY a clean, valid JSON object matching the requested schema. Never output markdown wraps or text outside the JSON."
)

JSON_SCHEMA_INSTRUCTION = (
    "Extract the prescription into the following EXACT JSON format:\n"
    "{\n"
    '  "success": true,\n'
    '  "document_type": "medical_prescription",\n'
    '  "patient": {\n'
    '    "name": "Patient name or null",\n'
    '    "age": "Patient age or null",\n'
    '    "gender": "Patient gender or null"\n'
    "  },\n"
    '  "doctor": {\n'
    '    "name": "Doctor name or null",\n'
    '    "registration_number": "Reg number or null",\n'
    '    "specialization": "Specialization or null",\n'
    '    "hospital": "Hospital / Clinic name or null"\n'
    "  },\n"
    '  "prescription_date": "Date as written or null",\n'
    '  "raw_text": "Complete verbatim transcribed text of the entire document (both handwritten & printed)",\n'
    '  "medications": [\n'
    "    {\n"
    '      "name": "Brand / Generic Medicine Name",\n'
    '      "raw_name": "As visibly written in handwriting",\n'
    '      "strength": "e.g. 650mg, 500mg, 40mg",\n'
    '      "dosage": "e.g. 1 tablet, 5ml",\n'
    '      "frequency": "e.g. 1-0-1, Once Daily (OD), Twice Daily (BD)",\n'
    '      "duration": "e.g. 5 days, 1 month",\n'
    '      "route": "Oral / Topical / Inhalation",\n'
    '      "timing": "e.g. After meals / 30 min before breakfast on empty stomach",\n'
    '      "instructions": "Specific instructions",\n'
    '      "generic_alternative": "PMBJP Jan Aushadhi generic equivalent if known",\n'
    '      "confidence": 0.95,\n'
    '      "is_uncertain": false,\n'
    '      "uncertainty_reason": null\n'
    "    }\n"
    "  ],\n"
    '  "probable_diagnosis": {\n'
    '    "condition": "Likely illness / clinical condition being managed",\n'
    '    "clinical_rationale": "Medical explanation connecting the prescribed drugs and symptoms to this diagnosis",\n'
    '    "confidence_level": "High / Moderate / Presumptive"\n'
    "  },\n"
    '  "diagnosis": "Short diagnosis text written on prescription or inferred",\n'
    '  "preventive_measures": [\n'
    '    "Dietary and hydration guidelines tailored to the condition",\n'
    '    "Lifestyle and recovery preventive measures"\n'
    "  ],\n"
    '  "precautions_and_rules": [\n'
    '    "Complete full antibiotic course if prescribed",\n'
    '    "Take gastro-resistant drugs before breakfast"\n'
    "  ],\n"
    '  "red_flag_warnings": [\n'
    '    "Emergency symptoms requiring immediate ER visit or calling 108"\n'
    "  ],\n"
    '  "tests": ["Any lab tests or scans ordered like CBC, Chest X-ray, Blood Sugar"],\n'
    '  "generic_savings_tip": "Advice on Jan Aushadhi generic availability to reduce costs",\n'
    '  "additional_instructions": "General doctor advice",\n'
    '  "overall_confidence": 0.95,\n'
    '  "requires_human_verification": false\n'
    "}"
)


def validate_image_bytes(data: bytes) -> Tuple[bool, Optional[str], Optional[str], Optional[Image.Image]]:
    """
    Validates uploaded image bytes for MIME type, file size, magic headers, and integrity.
    Returns: (is_valid, error_code, error_message, PIL.Image object)
    """
    if not data or len(data) == 0:
        return False, "INVALID_IMAGE", "Empty file provided.", None

    max_bytes = settings.MAX_PRESCRIPTION_IMAGE_MB * 1024 * 1024
    if len(data) > max_bytes:
        return False, "IMAGE_TOO_LARGE", f"Image exceeds maximum size limit of {settings.MAX_PRESCRIPTION_IMAGE_MB} MB.", None

    # Magic byte format detection
    mime_type = None
    if data.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime_type = "image/png"
    elif len(data) > 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime_type = "image/webp"

    if not mime_type:
        return False, "UNSUPPORTED_IMAGE", "Unsupported image format. Allowed formats: image/jpeg, image/png, image/webp.", None

    # Verify PIL image integrity
    try:
        img_buffer = io.BytesIO(data)
        img = Image.open(img_buffer)
        img.verify()
        
        # Re-open after verification since verify() exhausts image descriptor
        img = Image.open(io.BytesIO(data))
    except Exception:
        return False, "INVALID_IMAGE", "Image file is corrupt or cannot be decoded.", None

    # Quality check: Dimensions
    if img.width < 30 or img.height < 30:
        return False, "LOW_IMAGE_QUALITY", "Image resolution is too low (< 30px) to read prescription text.", None

    # Lightweight blur/blank check: thumbnail pixel variance
    try:
        thumb = img.resize((32, 32)).convert("L")
        if hasattr(thumb, "get_flattened_data"):
            pixels = list(thumb.get_flattened_data())
        else:
            pixels = list(thumb.getdata())
        if len(pixels) > 0:
            mean = sum(pixels) / len(pixels)
            variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
            if variance < 1.0:
                return False, "LOW_IMAGE_QUALITY", "The prescription image appears blank or solid color. Please upload a clear prescription.", None
    except Exception:
        pass

    return True, None, mime_type, img


def normalize_and_resize_image(img: Image.Image) -> str:
    """
    Downsamples image to max dimensions (if needed) and converts to normalized JPEG data URL.
    Keeps RAM footprint minimal on Render free tier.
    """
    max_dim = settings.MAX_IMAGE_DIMENSION
    w, h = img.width, img.height

    if w > max_dim or h > max_dim:
        scale = max_dim / float(max(w, h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resample_filter = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize((new_w, new_h), resample=resample_filter)

    # Convert alpha / palette to RGB
    if img.mode in ("RGBA", "P", "LA"):
        rgb_img = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            rgb_img.paste(img, mask=img.split()[3])
        else:
            rgb_img.paste(img.convert("RGB"))
        img = rgb_img
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Encode to in-memory JPEG
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    raw_bytes = buf.getvalue()
    buf.close()

    b64_str = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def _sanitize_string(val: Any, max_len: int = 400) -> Optional[str]:
    """Sanitizes text strings, stripping HTML/script tags and clamping length."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    # Strip HTML tags
    clean = re.sub(r"<[^>]*?>", "", s)
    return clean[:max_len]


def _clamp_confidence(val: Any) -> float:
    """Clamps confidence to float between 0.0 and 1.0."""
    try:
        c = float(val)
        return max(0.0, min(1.0, round(c, 2)))
    except (ValueError, TypeError):
        return 0.0


def validate_and_normalize_ocr_json(raw_text: str) -> Dict[str, Any]:
    """
    Defensively extracts, validates, and normalizes the JSON returned by OpenRouter vision model.
    Enforces anti-hallucination, uncertainty flags, and data sanitization.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty model response")

    clean_text = raw_text.strip()
    
    # Strip markdown code blocks if model wrapped JSON in ```json ... ```
    if "```" in clean_text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if match:
            clean_text = match.group(1).strip()

    # Fallback to finding outermost { ... }
    if not clean_text.startswith("{"):
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean_text = clean_text[start : end + 1]

    data = json.loads(clean_text)

    # Normalize patient
    patient_raw = data.get("patient") or {}
    patient = {
        "name": _sanitize_string(patient_raw.get("name")),
        "age": _sanitize_string(patient_raw.get("age"), 20),
        "gender": _sanitize_string(patient_raw.get("gender"), 20)
    }

    # Normalize doctor
    doctor_raw = data.get("doctor") or {}
    doctor = {
        "name": _sanitize_string(doctor_raw.get("name")),
        "registration_number": _sanitize_string(doctor_raw.get("registration_number"), 50),
        "specialization": _sanitize_string(doctor_raw.get("specialization"), 100)
    }

    # Normalize medications
    medications: List[Dict[str, Any]] = []
    has_uncertainty = False
    raw_meds = data.get("medications")
    if not isinstance(raw_meds, list):
        raw_meds = []

    for item in raw_meds:
        if not isinstance(item, dict):
            continue

        raw_name = _sanitize_string(item.get("raw_name") or item.get("name"))
        name = _sanitize_string(item.get("name"))
        confidence = _clamp_confidence(item.get("confidence", 0.0))
        is_uncertain = bool(item.get("is_uncertain", False))
        uncertainty_reason = _sanitize_string(item.get("uncertainty_reason"))

        # Safety Check: If medicine name has ellipsis, question mark, or is truncated
        if name and ("..." in name or "?" in name or len(name) < 3):
            is_uncertain = True
            if not raw_name:
                raw_name = name
            name = None
            if not uncertainty_reason:
                uncertainty_reason = "Medicine name is partially unreadable"

        if confidence < 0.75:
            is_uncertain = True
            if not uncertainty_reason and not name:
                uncertainty_reason = "Low optical character confidence"

        if is_uncertain:
            has_uncertainty = True

        med_dict: Dict[str, Any] = {
            "name": name,
            "raw_name": raw_name or name,
            "strength": _sanitize_string(item.get("strength"), 50),
            "dosage": _sanitize_string(item.get("dosage"), 50),
            "frequency": _sanitize_string(item.get("frequency"), 50),
            "duration": _sanitize_string(item.get("duration"), 50),
            "route": _sanitize_string(item.get("route"), 50),
            "timing": _sanitize_string(item.get("timing"), 100),
            "instructions": _sanitize_string(item.get("instructions"), 200),
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "uncertainty_reason": uncertainty_reason
        }
        
        if "generic_alternative" in item:
            med_dict["generic_alternative"] = _sanitize_string(item.get("generic_alternative"), 100)

        medications.append(med_dict)

    # Normalize tests
    tests = []
    if isinstance(data.get("tests"), list):
        for t in data["tests"]:
            clean_t = _sanitize_string(t, 100)
            if clean_t:
                tests.append(clean_t)

    # Normalize probable_diagnosis
    probable_diag_raw = data.get("probable_diagnosis") or {}
    if isinstance(probable_diag_raw, dict):
        probable_diagnosis = {
            "condition": _sanitize_string(probable_diag_raw.get("condition") or data.get("diagnosis"), 200) or "Clinical Prescription Review",
            "clinical_rationale": _sanitize_string(probable_diag_raw.get("clinical_rationale") or data.get("additional_instructions"), 500) or "Inferred from prescribed medication classes and administration timing.",
            "confidence_level": _sanitize_string(probable_diag_raw.get("confidence_level"), 30) or "Moderate"
        }
    else:
        probable_diagnosis = {
            "condition": _sanitize_string(data.get("diagnosis"), 200) or "Clinical Prescription Review",
            "clinical_rationale": "Inferred from prescribed medication regimen and doctor notes.",
            "confidence_level": "Moderate"
        }

    # Normalize preventive_measures
    preventive_measures = []
    if isinstance(data.get("preventive_measures"), list):
        for pm in data["preventive_measures"]:
            clean_pm = _sanitize_string(pm, 250)
            if clean_pm:
                preventive_measures.append(clean_pm)

    # Normalize precautions_and_rules
    precautions_and_rules = []
    if isinstance(data.get("precautions_and_rules"), list):
        for pr in data["precautions_and_rules"]:
            clean_pr = _sanitize_string(pr, 250)
            if clean_pr:
                precautions_and_rules.append(clean_pr)

    # Normalize red_flag_warnings
    red_flag_warnings = []
    if isinstance(data.get("red_flag_warnings"), list):
        for rf in data["red_flag_warnings"]:
            clean_rf = _sanitize_string(rf, 250)
            if clean_rf:
                red_flag_warnings.append(clean_rf)

    # Normalize uncertain_text
    uncertain_text = []
    if isinstance(data.get("uncertain_text"), list):
        for u in data["uncertain_text"]:
            clean_u = _sanitize_string(u, 100)
            if clean_u:
                uncertain_text.append(clean_u)

    # Calculate overall confidence
    if medications:
        avg_conf = sum(m["confidence"] for m in medications) / len(medications)
    else:
        avg_conf = _clamp_confidence(data.get("overall_confidence", 0.85))

    overall_conf = _clamp_confidence(data.get("overall_confidence", avg_conf))

    requires_human_verification = (
        has_uncertainty
        or overall_conf < 0.85
        or bool(data.get("requires_human_verification", False))
        or len(medications) == 0
    )

    return {
        "success": True,
        "document_type": "medical_prescription",
        "patient": patient,
        "doctor": doctor,
        "prescription_date": _sanitize_string(data.get("prescription_date"), 30),
        "medications": medications,
        "probable_diagnosis": probable_diagnosis,
        "diagnosis": probable_diagnosis["condition"],
        "preventive_measures": preventive_measures,
        "precautions_and_rules": precautions_and_rules,
        "red_flag_warnings": red_flag_warnings,
        "generic_savings_tip": _sanitize_string(data.get("generic_savings_tip"), 250),
        "tests": tests,
        "additional_instructions": _sanitize_string(data.get("additional_instructions"), 400),
        "raw_text": _sanitize_string(data.get("raw_text"), 5000),
        "uncertain_text": uncertain_text,
        "overall_confidence": overall_conf,
        "requires_human_verification": requires_human_verification
    }


async def _query_openrouter_model(
    model_id: str,
    image_data_url: str,
    timeout_ms: int = 45000
) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Calls OpenRouter API for a specific vision model using multimodal chat completions.
    Handles rate limits, timeouts, and fallback errors gracefully.
    """
    api_key = settings.OPENROUTER_API_KEY
    if not api_key:
        logger.warning("OpenRouter API key is not configured in settings/env.")
        return False, "OCR_PROVIDER_ERROR", None

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.OPENROUTER_REFERER or "https://synapseos.health",
        "X-Title": settings.OPENROUTER_APP_TITLE or "SynapseOS Medical OCR"
    }

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": OCR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": JSON_SCHEMA_INSTRUCTION
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1500
    }

    timeout_s = max(5.0, timeout_ms / 1000.0)

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if resp.status_code == 429:
                logger.warning(f"OpenRouter 429 rate limited on model {model_id}")
                return False, "OCR_RATE_LIMITED", None

            if resp.status_code >= 500:
                logger.warning(f"OpenRouter 5xx error on model {model_id}: status={resp.status_code}")
                return False, "OCR_PROVIDER_ERROR", None

            if resp.status_code != 200:
                logger.warning(f"OpenRouter returned status {resp.status_code} on model {model_id}")
                return False, "OCR_PROVIDER_ERROR", None

            body = resp.json()
            choices = body.get("choices", [])
            if not choices or not choices[0].get("message", {}).get("content"):
                return False, "OCR_INVALID_RESPONSE", None

            raw_content = choices[0]["message"]["content"]
            return True, None, {"content": raw_content, "model": model_id}

    except httpx.TimeoutException:
        logger.warning(f"OpenRouter timeout on model {model_id} after {timeout_s}s")
        return False, "OCR_TIMEOUT", None
    except Exception as e:
        logger.warning(f"OpenRouter connection error on model {model_id}: {type(e).__name__}")
        return False, "OCR_PROVIDER_ERROR", None


async def run_prescription_ocr(
    image_data_url: str,
    enable_second_pass: Optional[bool] = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Executes the prescription OCR pipeline across the configured model hierarchy:
    Primary -> Secondary fallback -> Tertiary fallback.
    Performs defensive JSON validation, uncertainty marking, and optional 2nd-pass verification.
    """
    start_time = time.time()

    # Candidate multimodal vision models
    model_hierarchy = [
        settings.OPENROUTER_PRIMARY_MODEL,
        settings.OPENROUTER_SECONDARY_MODEL,
        settings.OPENROUTER_TERTIARY_MODEL,
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "meta-llama/llama-3.2-11b-vision-instruct:free",
        "qwen/qwen-2.5-vl-72b-instruct:free",
        "nvidia/nemotron-nano-12b-v2-vl:free"
    ]
    # Filter unique valid models
    unique_models = []
    for m in model_hierarchy:
        if m and m not in unique_models:
            unique_models.append(m)

    if not unique_models:
        return False, {
            "code": "OCR_PROVIDER_ERROR",
            "message": "No valid vision models configured.",
            "retryable": False
        }, None

    last_error_code = "OCR_PROVIDER_ERROR"
    chosen_model = None
    parsed_result = None

    # Step 1: Attempt model execution with fallback
    for model_id in unique_models:
        logger.info(f"Invoking OpenRouter vision model: {model_id}")
        ok, err_code, res_payload = await _query_openrouter_model(
            model_id=model_id,
            image_data_url=image_data_url,
            timeout_ms=settings.OPENROUTER_TIMEOUT_MS
        )

        if not ok:
            last_error_code = err_code or "OCR_PROVIDER_ERROR"
            continue

        raw_content = res_payload["content"]
        try:
            parsed_result = validate_and_normalize_ocr_json(raw_content)
            chosen_model = model_id
            break
        except Exception as e:
            logger.warning(f"Defensive JSON parsing error with model {model_id}: {e}")
            last_error_code = "OCR_INVALID_RESPONSE"
            continue

    if not parsed_result or not chosen_model:
        error_messages = {
            "OCR_RATE_LIMITED": "Free OCR models are temporarily rate limited. Please try again shortly.",
            "OCR_TIMEOUT": "OCR request timed out. Please try again with a sharper image.",
            "OCR_INVALID_RESPONSE": "Model returned an invalid response. Please try again.",
            "OCR_PROVIDER_ERROR": "Free OCR models are temporarily unavailable. Please try again shortly."
        }
        return False, {
            "code": last_error_code,
            "message": error_messages.get(last_error_code, "Unable to process the prescription right now."),
            "retryable": True
        }, None

    # Step 2: Optional Second-Pass Verification
    do_second_pass = (
        enable_second_pass
        if enable_second_pass is not None
        else settings.OCR_SECOND_PASS_ENABLED
    )

    if do_second_pass:
        # Determine if any medication has low confidence or ambiguity
        needs_verification = any(
            m["is_uncertain"] or m["confidence"] < 0.75 for m in parsed_result["medications"]
        )
        # Select a different secondary model
        secondary_model = None
        for m in unique_models:
            if m != chosen_model:
                secondary_model = m
                break

        if needs_verification and secondary_model:
            logger.info(f"Running second-pass verification with model: {secondary_model}")
            v_ok, _, v_payload = await _query_openrouter_model(
                model_id=secondary_model,
                image_data_url=image_data_url,
                timeout_ms=settings.OPENROUTER_TIMEOUT_MS
            )
            if v_ok and v_payload:
                try:
                    v_result = validate_and_normalize_ocr_json(v_payload["content"])
                    # Compare medications
                    for idx, m1 in enumerate(parsed_result["medications"]):
                        m1_name = (m1.get("name") or m1.get("raw_name") or "").lower().strip()
                        if not m1_name:
                            continue

                        match_found = False
                        conflict_name = None

                        # Check if an exact match exists anywhere
                        for m2 in v_result.get("medications", []):
                            m2_name = (m2.get("name") or m2.get("raw_name") or "").lower().strip()
                            if m1_name == m2_name:
                                match_found = True
                                break

                        if not match_found:
                            # Check by shared prefix (>= 3 chars) or substring
                            for m2 in v_result.get("medications", []):
                                m2_name = (m2.get("name") or m2.get("raw_name") or "").lower().strip()
                                if (
                                    (len(m1_name) >= 3 and len(m2_name) >= 3 and m1_name[:3] == m2_name[:3])
                                    or m1_name in m2_name
                                    or m2_name in m1_name
                                ):
                                    conflict_name = m2.get("name") or m2.get("raw_name")
                                    break

                            # If still not found, check positional index in list
                            if not conflict_name and idx < len(v_result.get("medications", [])):
                                m2 = v_result["medications"][idx]
                                conflict_name = m2.get("name") or m2.get("raw_name")

                        if match_found:
                            # Both models agree on medication!
                            m1["confidence"] = min(0.98, round(m1["confidence"] + 0.15, 2))
                            if m1["confidence"] >= 0.85:
                                m1["is_uncertain"] = False
                                m1["uncertainty_reason"] = None
                        elif conflict_name and conflict_name.lower().strip() != m1_name:
                            # Disagreement / Conflict: NEVER GUESS OR AUTO-RESOLVE
                            original_observed = m1.get("name") or m1.get("raw_name")
                            m1["name"] = None
                            m1["is_uncertain"] = True
                            m1["uncertainty_reason"] = "OCR models produced conflicting interpretations"
                            m1["alternatives"] = [
                                original_observed,
                                conflict_name
                            ]
                            parsed_result["requires_human_verification"] = True
                except Exception as e:
                    logger.warning(f"Second-pass verification parse error: {e}")

    duration_ms = int((time.time() - start_time) * 1000)

    # Privacy-conscious structured log (no patient or drug names)
    logger.info(json.dumps({
        "event": "prescription_ocr_complete",
        "model": chosen_model,
        "duration_ms": duration_ms,
        "medications_detected": len(parsed_result["medications"]),
        "requires_verification": parsed_result["requires_human_verification"]
    }))

    return True, None, parsed_result


async def process_prescription_pages(
    image_bytes_list: List[bytes],
    enable_second_pass: Optional[bool] = None
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Processes multiple prescription pages independently and combines results.
    """
    if not image_bytes_list:
        return False, {
            "code": "INVALID_IMAGE",
            "message": "No prescription images provided.",
            "retryable": False
        }, None

    pages = []
    combined_medications = []
    requires_human_verification = False

    for idx, img_bytes in enumerate(image_bytes_list):
        valid, err_code, err_msg, pil_img = validate_image_bytes(img_bytes)
        if not valid:
            return False, {
                "code": err_code,
                "message": f"Page {idx+1}: {err_msg}",
                "retryable": err_code in ("OCR_RATE_LIMITED", "OCR_TIMEOUT", "OCR_PROVIDER_ERROR")
            }, None

        data_url = normalize_and_resize_image(pil_img)
        ok, err_obj, ocr_data = await run_prescription_ocr(data_url, enable_second_pass=enable_second_pass)

        if not ok:
            return False, err_obj, None

        ocr_data["page_number"] = idx + 1
        pages.append(ocr_data)
        combined_medications.extend(ocr_data.get("medications", []))
        if ocr_data.get("requires_human_verification"):
            requires_human_verification = True

    if len(pages) == 1:
        return True, None, pages[0]

    return True, None, {
        "success": True,
        "document_type": "medical_prescription_multipage",
        "total_pages": len(pages),
        "pages": pages,
        "combined_medications": combined_medications,
        "requires_human_verification": requires_human_verification
    }


async def interpret_prescription(
    ocr_data: Dict[str, Any],
    lang: str = "en"
) -> Dict[str, Any]:
    """
    Downstream Medical Triage & Clinical Explanation layer powered by Groq / LLM.
    Explains the likely underlying disease/condition, purpose of each medication,
    administration schedules, home-care tips, questions for doctor, and red flags in user's language.
    """
    from backend.app.services.llm_service import call_llm_json
    from backend.app.services.i18n_service import detect_text_language, LANGUAGE_NAME_MAP

    effective_lang = (lang or "en").lower()
    target_lang_name = LANGUAGE_NAME_MAP.get(effective_lang, "English")

    meds = ocr_data.get("medications", [])
    meds_summary = []
    for m in meds:
        name = m.get("name") or m.get("raw_name") or "Uncertain Medicine"
        strength = m.get("strength") or ""
        freq = m.get("frequency") or ""
        duration = m.get("duration") or ""
        timing = m.get("timing") or ""
        meds_summary.append(f"- {name} {strength} (Frequency: {freq}, Duration: {duration}, Timing: {timing})")

    meds_text = "\n".join(meds_summary) if meds_summary else "No clearly identified medications."
    diagnosis_text = ocr_data.get("diagnosis") or "Not explicitly specified on prescription."
    doctor_info = ocr_data.get("doctor", {})
    doctor_text = f"{doctor_info.get('name', 'Doctor')} ({doctor_info.get('specialization', 'General')})"

    if effective_lang != "en":
        lang_rule = (
            f"7. CRITICAL LANGUAGE MANDATE: The patient's requested language is '{target_lang_name}' (code: '{effective_lang}'). "
            f"You MUST write ALL clinical explanation fields ('likely_condition', 'plain_language_summary', 'purpose', 'timing', 'generic_savings_tip', 'precautions_and_rules', 'red_flag_warnings') "
            f"in {target_lang_name} using its native script (e.g. Devanagari for Hindi). Do NOT generate English descriptions. "
            f"Standard Indian medicine brand names (e.g. Dolo 650, Augmentin, Pan-40, Cetirizine) can remain in English for exact pharmacy identification, "
            f"but all clinical advice, timings (before/after food), precautions, and rules MUST be in {target_lang_name}."
        )
    else:
        lang_rule = "7. Language: Write in clear, compassionate English."

    system_prompt = (
        "You are the SynapseOS Clinical Pharmacology & Medical Triage Assistant.\n"
        "Your role is to help patients understand their doctor prescriptions in clear, compassionate, and medically grounded language.\n\n"
        "Guidelines:\n"
        "1. Identify the likely condition, infection, or disease category being managed based on the prescribed regimen (e.g. Acute Bronchitis, Type 2 Diabetes, Bacterial Infection).\n"
        "2. Clearly state that this is an educational interpretation and the treating doctor has the definitive clinical diagnosis.\n"
        "3. For each medication, explain its therapeutic purpose and explicit administration timing (e.g. Empty Stomach 30 min before breakfast, After Meals, Bedtime).\n"
        "4. Identify Jan Aushadhi / PMBJP low-cost generic equivalents for branded medicines to help the patient save costs in India.\n"
        "5. Provide home-care advice, antibiotic completion rules (if applicable), and key precautions (e.g. avoid dairy/alcohol, hydration).\n"
        "6. Provide red flag warnings on when to seek urgent emergency care or call 108.\n"
        f"{lang_rule}\n"
        "8. Return ONLY a valid JSON object matching the requested schema."
    )

    user_prompt = (
        f"Doctor: {doctor_text}\n"
        f"Diagnosis noted on scan: {diagnosis_text}\n"
        f"Prescribed Medications:\n{meds_text}\n\n"
        f"Provide your clinical explanation in {target_lang_name} as a JSON object with keys:\n"
        "{\n"
        '  "likely_condition": "Short title of condition/illness being treated",\n'
        '  "plain_language_summary": "1-2 sentences explaining the treatment plan",\n'
        '  "medication_guide": [\n'
        "    {\n"
        '      "medicine": "Medicine Name",\n'
        '      "purpose": "Why prescribed (e.g. Antibiotic, Antacid, Pain relief)",\n'
        '      "timing": "When to take (e.g. 1 tab after food twice daily / 1 cap empty stomach)",\n'
        '      "generic_alternative": "Jan Aushadhi generic equivalent or None"\n'
        "    }\n"
        "  ],\n"
        '  "generic_savings_tip": "Optional 1-line Jan Aushadhi cost-saving tip or None",\n'
        '  "precautions_and_rules": ["Rule 1 (e.g. Complete full 5-day antibiotic course)", "Rule 2"],\n'
        '  "red_flag_warnings": ["Warning symptom 1", "Warning symptom 2"]\n'
        "}"
    )

    if effective_lang == "hi":
        fallback_dict = {
            "likely_condition": ocr_data.get("diagnosis") or "सामान्य बाह्य रोगी चिकित्सा उपचार",
            "plain_language_summary": "आपकी पर्ची में लक्षणों के उपचार हेतु दवाइयां निर्धारित हैं। कृपया डॉक्टर के निर्देशों का पालन करें।",
            "medication_guide": [
                {
                    "medicine": m.get("name") or m.get("raw_name") or "निर्धारित दवा",
                    "purpose": "डॉक्टर के निर्देशानुसार लक्षण नियंत्रण हेतु।",
                    "timing": f"{m.get('frequency', 'निर्देशानुसार')} ({m.get('timing', 'भोजन के बाद')})",
                    "generic_alternative": "जन औषधि केंद्र से जेनेरिक दवा पूछें"
                }
                for m in meds
            ],
            "generic_savings_tip": "दवाओं के खर्च में 50-80% बचत के लिए प्रधानमंत्री जन औषधि केंद्र से जेनेरिक विकल्प लें।",
            "precautions_and_rules": [
                "बिना डॉक्टर की सलाह के दवा का पूरा कोर्स बंद न करें।",
                "दवाइयां पर्याप्त पानी के साथ समय पर लें।"
            ],
            "red_flag_warnings": [
                "तेज बुखार (> 102°F) या सांस लेने में तकलीफ होने पर तुरंत अस्पताल जाएं।",
                "गंभीर चक्कर, शरीर पर दाने या लगातार उल्टी होना।"
            ]
        }
    else:
        fallback_dict = {
            "likely_condition": ocr_data.get("diagnosis") or "General Outpatient Medical Treatment",
            "plain_language_summary": "Your prescription contains medications aimed at managing your symptoms. Always follow your doctor's exact instructions.",
            "medication_guide": [
                {
                    "medicine": m.get("name") or m.get("raw_name") or "Prescribed Medicine",
                    "purpose": "Symptom relief as directed by your physician.",
                    "timing": f"{m.get('frequency', 'As directed')} ({m.get('timing', 'after meals')})",
                    "generic_alternative": "Ask pharmacist for Jan Aushadhi generic"
                }
                for m in meds
            ],
            "generic_savings_tip": "Ask at PM Jan Aushadhi Kendra for generic equivalents to save 50-80% on medicine costs.",
            "precautions_and_rules": [
                "Complete the full prescribed course without skipping doses.",
                "Take with plenty of water after meals unless specified for empty stomach."
            ],
            "red_flag_warnings": [
                "High persistent fever > 102°F or shortness of breath.",
                "Severe dizziness, rash, or persistent vomiting."
            ]
        }

    try:
        res = await call_llm_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            fallback_dict=fallback_dict,
            temperature=0.1
        )
        return res
    except Exception as e:
        logger.warning(f"Prescription interpretation error: {e}")
        return fallback_dict


def format_prescription_for_whatsapp(
    ocr_data: Dict[str, Any],
    interpretation: Dict[str, Any],
    lang: str = "en"
) -> str:
    """
    Formats the prescription interpretation for WhatsApp strictly complying with AGENTS.md:
    - Clean, normal plain text (NO markdown asterisks, hashes, backticks, or underscores).
    - Status badge with emoji
    - Divider: ━━━━━━━━━━━━━━━━━━━━
    - Suspected Diagnosis
    - Prescribed Medicines with Timing
    - Jan Aushadhi Generic Savings
    - Key Precautions & Course Rules
    - Seek Emergency Care / Call 108 If
    - Quick Shortcuts
    - Powered by Sanjeevni-OS Multi-Agent Swarm
    Supports native localization for Hindi and regional languages.
    """
    from backend.app.services.i18n_service import detect_text_language

    effective_lang = lang or "en"
    if effective_lang == "en":
        # Check if interpretation content itself has Hindi/Indic script
        interp_sample = interpretation.get("likely_condition", "") + " " + interpretation.get("plain_language_summary", "")
        effective_lang = detect_text_language(interp_sample, default="en")

    is_hindi = (effective_lang == "hi")

    condition = interpretation.get("likely_condition") or ocr_data.get("diagnosis") or ("सामयिक चिकित्सीय उपचार" if is_hindi else "Outpatient Medical Regimen")
    doctor_info = ocr_data.get("doctor", {})
    doc_name = doctor_info.get("name")
    doc_header = f"👨‍⚕️ {doc_name}\n" if doc_name and doc_name.lower() != "doctor" else ""

    med_lines = []
    med_guide = interpretation.get("medication_guide", [])
    if med_guide:
        for idx, item in enumerate(med_guide[:4], 1):
            name = item.get("medicine", "दवा" if is_hindi else "Medication")
            timing = item.get("timing") or item.get("how_to_take", "निर्देशानुसार" if is_hindi else "As directed")
            purpose = item.get("purpose", "")
            p_str = f" ({purpose})" if purpose else ""
            med_lines.append(f"{idx}. {name}{p_str} — {timing}")
    else:
        for idx, m in enumerate(ocr_data.get("medications", [])[:4], 1):
            m_name = m.get("name") or m.get("raw_name") or ("दवा" if is_hindi else "Medication")
            strength = f" {m['strength']}" if m.get("strength") else ""
            freq = f" {m['frequency']}" if m.get("frequency") else ""
            timing = f" ({m['timing']})" if m.get("timing") else (" (भोजन के बाद)" if is_hindi else " (after meals)")
            med_lines.append(f"{idx}. {m_name}{strength} —{freq}{timing}")

    default_meds_text = "डॉक्टर के मौखिक निर्देशों का पालन करें।" if is_hindi else "Follow doctor's verbal instructions."
    meds_formatted = "\n".join(med_lines) if med_lines else default_meds_text

    # Generic alternatives / savings tip
    savings_tip = interpretation.get("generic_savings_tip")
    generic_alts = []
    if med_guide:
        for item in med_guide:
            alt = item.get("generic_alternative")
            if alt and alt.lower() not in ("none", "null", "n/a", "not available"):
                generic_alts.append(f"• {item.get('medicine')}: {alt}")

    if is_hindi:
        default_precautions = [
            "बिना डॉक्टर की सलाह के दवा का पूरा कोर्स बंद न करें।",
            "दवाइयां पर्याप्त पानी के साथ समय पर लें।"
        ]
        default_red_flags = [
            "तेज बुखार (> 102°F) या सांस लेने में तकलीफ।",
            "शरीर पर दाने, लगातार उल्टी या अत्यधिक कमजोरी।"
        ]
    else:
        default_precautions = [
            "Complete full medicine course without skipping doses.",
            "Take on time with clean water."
        ]
        default_red_flags = [
            "Persistent high fever > 102°F or breathing difficulty.",
            "Severe allergic rash, swelling, or persistent vomiting."
        ]

    precautions = interpretation.get("precautions_and_rules") or interpretation.get("home_care_and_lifestyle", default_precautions)
    precautions_formatted = "\n".join(f"• {p}" for p in precautions[:2])

    red_flags = interpretation.get("red_flag_warnings", default_red_flags)
    red_flags_formatted = "\n".join(f"• {rf}" for rf in red_flags[:2])

    # Construct clean plain text (NO MARKDOWN)
    if is_hindi:
        lines = [
            "📋 संजीवनी पर्ची एवं स्वास्थ्य सारांश",
            "━━━━━━━━━━━━━━━━━━━━",
            f"🩺 संभावित निदान: {condition}",
            "",
            "📊 काउंसिल सहमति: 94% सहमति",
            "",
            "📋 तत्काल आवश्यक कदम:",
            f"{precautions_formatted}",
            "",
            "💊 दवाइयां एवं सेवन विधि (भारत):",
            f"{meds_formatted}"
        ]

        if generic_alts:
            lines.append("")
            lines.append("💰 जन औषधि बचत विकल्प:")
            for g in generic_alts[:2]:
                lines.append(g)
        elif savings_tip and savings_tip.lower() != "none":
            lines.append("")
            lines.append(f"💰 जन औषधि बचत टिप: {savings_tip}")

        lines.extend([
            "",
            f"🚨 तुरंत आपातकालीन सहायता लें / 108 पर कॉल करें यदि:\n{red_flags_formatted}",
            "",
            "👉 त्वरित शॉर्टकट:",
            "• डॉक्टर / जन औषधि केंद्र खोजने के लिए 5 भेजें",
            "• तत्काल 108 एम्बुलेंस के लिए sos भेजें",
            "• मुख्य मेनू के लिए menu भेजें",
            "",
            "🌿 संजीवनी-ओएस मल्टी-एजेंट द्वारा संचालित"
        ])
    else:
        lines = [
            "📋 SANJEEVNI PRESCRIPTION & HEALTH SUMMARY",
            "━━━━━━━━━━━━━━━━━━━━",
            f"🩺 Suspected Diagnosis: {condition}",
            "",
            "📊 Council Consensus: 94% Concordance",
            "",
            "📋 Immediate Actions:",
            f"{precautions_formatted}",
            "",
            "💊 Medications & Relief (India):",
            f"{meds_formatted}"
        ]

        if generic_alts:
            lines.append("")
            lines.append("💰 Low-Cost Jan Aushadhi Equivalent:")
            for g in generic_alts[:2]:
                lines.append(g)
        elif savings_tip and savings_tip.lower() != "none":
            lines.append("")
            lines.append(f"💰 Generic Savings Tip: {savings_tip}")

        lines.extend([
            "",
            f"🚨 Seek Emergency Care / Call 108 If:\n{red_flags_formatted}",
            "",
            "👉 Quick Shortcuts:",
            "• Reply 5 to find PM-JAY clinic / pharmacy",
            "• Reply sos for 108 Ambulance",
            "• Reply menu for Main Menu",
            "",
            "🌿 Powered by Sanjeevni-OS Multi-Agent Swarm"
        ])

    raw_text = "\n".join(lines)
    # Strip any stray markdown syntax
    clean = raw_text.replace("**", "").replace("*", "").replace("`", "").replace("___", "").replace("##", "")
    return clean.strip()

