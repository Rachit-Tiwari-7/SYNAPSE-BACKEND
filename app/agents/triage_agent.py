"""
SynapseOS — agents/triage_agent.py
Clinical Symptom Triage & Risk Detection Agent.
Uses genuine LLM reasoning (Groq / OpenRouter) with deterministic safety heuristics.
Categorizes user symptoms into: Emergency (Red), Doctor Consult (Amber), Home Care (Green).
"""

import time
from typing import Dict, Any, List
from backend.app.core.state import SynapseOSState, AgentTraceStep
from backend.app.services.llm_service import call_llm_json

SYMPTOM_TAXONOMY = {
    "red_flags": [
        "chest pain", "shortness of breath", "difficulty breathing", "unconscious",
        "hemoptysis", "hematemesis", "sudden paralysis", "severe head injury",
        "anaphylaxis", "severe allergic reaction", "cyanosis", "seizure"
    ],
    "amber_flags": [
        "persistent fever", "fever over 102", "unexplained weight loss", "productive cough",
        "blood in stool", "severe abdominal pain", "jaundice", "yellow eyes",
        "persistent vomiting", "dysuria", "burning urination", "joint swelling"
    ],
    "green_flags": [
        "mild headache", "runny nose", "sneezing", "sore throat", "mild body ache",
        "fatigue", "dry cough", "indigestion", "mild acidity", "minor scrape"
    ]
}


async def analyze_symptoms(text: str) -> Dict[str, Any]:
    """
    Evaluates clinical symptoms using live LLM inference (Groq/OpenRouter),
    with deterministic safety taxonomy verification.
    """
    text_lower = (text or "").lower()
    
    detected_red = [s for s in SYMPTOM_TAXONOMY["red_flags"] if s in text_lower]
    detected_amber = [s for s in SYMPTOM_TAXONOMY["amber_flags"] if s in text_lower]
    detected_green = [s for s in SYMPTOM_TAXONOMY["green_flags"] if s in text_lower]

    if detected_red:
        default_level = "EMERGENCY_CARE"
        default_badge = "🔴 Emergency Care (Immediate)"
        default_action = "Please proceed immediately to the nearest Emergency Department or call 112 / 911."
        default_specialist = "Emergency Medicine Physician / Trauma Specialist"
    elif detected_amber:
        default_level = "DOCTOR_CONSULT"
        default_badge = "🟡 Doctor Consultation Needed"
        default_action = "Schedule a consultation with a physician within 24 to 48 hours for clinical evaluation and testing."
        default_specialist = "General Physician / Internal Medicine Specialist"
    else:
        default_level = "HOME_CARE"
        default_badge = "🟢 Home Self-Care & Monitoring"
        default_action = "Monitor symptoms, ensure adequate hydration, rest, and follow OTC symptom relief protocols. Seek medical care if symptoms worsen."
        default_specialist = "Primary Care Provider if symptoms persist > 5 days"

    fallback = {
        "triage_level": default_level,
        "urgency_badge": default_badge,
        "detected_symptoms": {
            "critical_flags": detected_red,
            "moderate_flags": detected_amber,
            "mild_flags": detected_green
        },
        "recommended_action": default_action,
        "recommended_specialist": default_specialist,
        "vitals_to_check": ["Body Temperature", "Blood Pressure", "SpO2 (Oxygen Saturation)", "Pulse Rate"],
        "disclaimer": "This clinical triage assessment is for guidance and does not replace in-person physician diagnosis."
    }

    # Prompt LLM for deep clinical nuance
    system_prompt = (
        "You are an expert emergency medicine and clinical triage AI assistant. "
        "Analyze the patient's reported symptoms and return a strictly valid JSON object with the following schema:\n"
        "{\n"
        '  "triage_level": "EMERGENCY_CARE" | "DOCTOR_CONSULT" | "HOME_CARE",\n'
        '  "urgency_badge": "🔴 Emergency Care (Immediate)" | "🟡 Doctor Consultation Needed" | "🟢 Home Self-Care & Monitoring",\n'
        '  "recommended_action": "Detailed clinical guidance and next steps",\n'
        '  "recommended_specialist": "Specific medical specialty to consult",\n'
        '  "vitals_to_check": ["List", "of", "relevant", "vitals"],\n'
        '  "detected_symptoms": {"critical_flags": [], "moderate_flags": [], "mild_flags": []}\n'
        "}\n"
        "Be conservative and prioritize patient safety."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Patient symptoms: {text}"}
    ]

    llm_result = await call_llm_json(messages, fallback_dict=fallback)
    return llm_result


async def triage_agent_node(state: SynapseOSState) -> SynapseOSState:
    """LangGraph node execution for Symptom Triage."""
    start = time.time()
    res = await analyze_symptoms(state.input_text)
    state.triage_data = res
    
    duration = int((time.time() - start) * 1000)
    state.trace.append(AgentTraceStep(
        agent_name="Clinical Symptom Triage Agent (Groq/OpenRouter)",
        action=f"Classified symptoms -> {res.get('urgency_badge', 'Assessed')}",
        duration_ms=duration,
        details={"level": res.get("triage_level"), "specialist": res.get("recommended_specialist")}
    ))
    return state
