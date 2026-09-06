"""
SynapseOS — agents/orchestrator.py
Central Multi-Agent Swarm Orchestrator & StateGraph Pipeline.
Coordinates Safety Gate -> Intent Routing -> Specialist Agents (Triage, Drug, Scan, Mental) -> AI Council -> Unified LLM Synthesis.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional
from backend.app.core.state import SynapseOSState, AgentTraceStep
from backend.app.core.safety_router import evaluate_safety
from backend.app.agents.drug_agent import drug_agent_node
from backend.app.agents.triage_agent import triage_agent_node
from backend.app.agents.verification_agent import verification_agent_node
from backend.app.agents.scan_agent import scan_agent_node
from backend.app.agents.mental_health_agent import mental_health_node
from backend.app.agents.vaccination_agent import vaccination_agent_node
from backend.app.agents.preventive_health_agent import preventive_health_agent_node
from backend.app.agents.outbreak_agent import outbreak_agent_node
from backend.app.ml.digital_twin import compute_baseline_organ_scores, DigitalTwinInput
from backend.app.services.llm_service import call_llm


def detect_intent(text: str) -> str:
    """Classifies user query intent."""
    text_lower = (text or "").lower()
    
    if any(k in text_lower for k in ["vaccin", "uip", "u-win", "immuniz", "polio", "bcg", "pentavalent", "booster dose", "child dose"]):
        return "VACCINATION_SCHEDULE"
    elif any(k in text_lower for k in ["outbreak", "epidemic", "dengue case", "malaria surge", "cholera", "nipah", "surveillance", "hotspot"]):
        return "OUTBREAK_ALERT"
    elif any(k in text_lower for k in ["ors", "prevent", "poshan", "nutrition", "breastfeed", "anemia", "clean water", "hygiene", "mosquito net", "awareness quiz"]):
        return "PREVENTIVE_HEALTH"
    elif any(k in text_lower for k in ["xray", "x-ray", "fracture", "bone", "mri", "scan", "prescription", "report"]):
        return "SCAN_ANALYSIS"
    elif any(k in text_lower for k in ["take with", "interact", "drug", "medicine", "pill", "paracetamol", "aspirin", "dosage", "ibuprofen"]):
        return "DRUG_SAFETY"
    elif any(k in text_lower for k in ["stress", "anxious", "anxiety", "depressed", "period", "cramp", "menstrual", "sad", "hopeless"]):
        return "MENTAL_HEALTH"
    elif any(k in text_lower for k in ["body", "organs", "twin", "vitality", "health score"]):
        return "DIGITAL_TWIN"
    else:
        return "SYMPTOM_TRIAGE"


from backend.app.services.i18n_service import detect_text_language, LANGUAGE_NAME_MAP


async def orchestrate_health_request(
    message: str,
    channel: str = "web",
    session_id: str = None,
    user_id: str = "demo_user",
    language: Optional[str] = None
) -> SynapseOSState:
    """
    Executes the full multi-agent DAG workflow for any user message.
    Optimized with concurrent asyncio.gather multi-agent execution and channel-tuned token synthesis.
    """
    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    # Auto-detect language if not explicitly provided or default 'en'
    if not language or language == "en":
        effective_lang = detect_text_language(message, default="en")
    else:
        effective_lang = language.lower()

    state = SynapseOSState(
        session_id=session_id,
        user_id=user_id,
        channel=channel,
        input_text=message,
        language=effective_lang
    )

    # 1. Deterministic Safety Gate Check
    start_time = time.time()
    safety = evaluate_safety(message)
    if not safety.is_safe:
        state.safety_cleared = False
        state.safety_message = safety.response
        state.final_response = safety.response
        state.trace.append(AgentTraceStep(
            agent_name="Deterministic Safety Gate",
            action=f"🚨 Immediate Emergency/Crisis Flag Intercepted ({safety.category})",
            duration_ms=int((time.time() - start_time) * 1000),
            details={"category": safety.category}
        ))
        return state

    state.trace.append(AgentTraceStep(
        agent_name="Deterministic Safety Gate",
        action="Passed safety verification protocol",
        duration_ms=int((time.time() - start_time) * 1000)
    ))

    # 2. Intent Routing
    intent = detect_intent(message)
    state.detected_intent = intent

    # 3. Dynamic Parallel Multi-Agent Execution based on Intent (Ultra-Fast Concurrent asyncio.gather)
    if intent == "VACCINATION_SCHEDULE":
        await asyncio.gather(vaccination_agent_node(state), verification_agent_node(state), return_exceptions=True)
    elif intent == "PREVENTIVE_HEALTH":
        await asyncio.gather(preventive_health_agent_node(state), verification_agent_node(state), return_exceptions=True)
    elif intent == "OUTBREAK_ALERT":
        await asyncio.gather(outbreak_agent_node(state), verification_agent_node(state), return_exceptions=True)
    elif intent == "DRUG_SAFETY":
        await asyncio.gather(drug_agent_node(state), triage_agent_node(state), verification_agent_node(state), return_exceptions=True)
    elif intent == "SCAN_ANALYSIS":
        await asyncio.gather(scan_agent_node(state), triage_agent_node(state), verification_agent_node(state), return_exceptions=True)
    elif intent == "MENTAL_HEALTH":
        await asyncio.gather(mental_health_node(state), triage_agent_node(state), return_exceptions=True)
    elif intent == "DIGITAL_TWIN":
        twin_data = compute_baseline_organ_scores(DigitalTwinInput())
        state.digital_twin = twin_data
        state.trace.append(AgentTraceStep(
            agent_name="3D Digital Health Twin Engine",
            action=f"Computed multi-organ vitality index ({twin_data['overall_health_score']}/100)",
            duration_ms=15
        ))
    else:
        # Default Full Swarm Consultation: Concurrent Triage + Drug + AI Council in Parallel
        await asyncio.gather(triage_agent_node(state), drug_agent_node(state), verification_agent_node(state), return_exceptions=True)

    # 4. Synthesize Final Consolidated Response via LLM (Groq / OpenRouter)
    synth_start = time.time()
    target_lang_name = LANGUAGE_NAME_MAP.get(effective_lang, "English")
    
    if effective_lang != "en":
        language_rule = (
            f"4. CRITICAL MANDATORY LANGUAGE: The user's query is in {target_lang_name} (code: '{effective_lang}'). "
            f"You MUST write the ENTIRE clinical response in {target_lang_name} using its native script (e.g., Devanagari for Hindi). "
            f"Never respond in English when {target_lang_name} is requested or used. Standard Indian medicine names (e.g. Dolo 650, Electral ORS, Pan-40, Cetirizine) "
            f"and helpline numbers (112, 108) may remain in Latin or native script, but all instructions, warnings, headers, and advice MUST be in {target_lang_name}."
        )
    else:
        language_rule = "4. Language: Respond in clear, accessible English."

    system_prompt = (
        "You are SynapseOS AI, an intelligent, empathetic, direct medical assistant for Indian healthcare.\n\n"
        "STRICT RULES FOR YOUR RESPONSE:\n"
        "1. BE SHORT, SIMPLE, AND TO THE POINT (under 130-160 words). Never use corporate filler, repetitive preamble, or robotic meta-talk.\n"
        "2. When the user reports symptoms (e.g. fever, headache, vomiting, cold, stomach ache):\n"
        "   - Probable Condition: State the likely clinical impression in the first sentence.\n"
        "   - Probable Indian Medications & Usage: List 1-2 standard Indian OTC medicines with precise timing (e.g., Dolo 650 / Paracetamol 650mg: 1 tab after meals with water, max 3/day; Electral ORS for hydration; Pan-40 before breakfast on empty stomach; Cetirizine at bedtime). If severe emergency, advise withholding self-medication until doctor exam.\n"
        "   - Next Steps: State when to see a General Physician / visit clinic (e.g. if symptoms persist > 48h).\n"
        "   - Preventive Measures & Care: Hydration, rest, light diet, sponge baths for fever.\n"
        "   - Red Flags: Rush to emergency or call 108/112 if SpO2 < 92%, difficulty breathing, fever > 103°F, neck stiffness.\n"
        "   - AI Disclaimer: Always end with a brief disclaimer: '⚠️ AI Recommendation: Educational guidance only. Please consult a qualified doctor or clinic for diagnosis and prescription.'\n"
        "3. If asking about a medicine (e.g. 'what is Calpol'): State what it is, typical usage, and safety precautions.\n"
        "4. If acute emergency (chest pain, stroke, severe trauma): Immediately instruct to call 108/112 or visit nearest emergency room.\n"
        f"{language_rule}"
    )
    
    agent_findings_context = f"""
Patient Query: {message}
Language: {target_lang_name} ({effective_lang})
Vaccination Status: {state.vaccination_data}
Preventive Health Data: {state.preventive_data}
Outbreak Surveillance: {state.outbreak_data}
Triage Data: {state.triage_data}
Drug Safety: {state.drug_check}
Scan Analysis: {state.scan_analysis}
AI Council Verification: {state.verification}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Consolidate these specialist agent findings for the patient in {target_lang_name}:\n{agent_findings_context}"}
    ]

    max_synth_tokens = 320 if channel in ("whatsapp", "sms") else 650
    llm_synthesis = await call_llm(messages, temperature=0.2, max_tokens=max_synth_tokens)

    if llm_synthesis:
        state.final_response = llm_synthesis
        state.trace.append(AgentTraceStep(
            agent_name="Swarm Synthesis & Reasoning Engine (Groq/OpenRouter)",
            action=f"Synthesized multi-agent findings in {target_lang_name}",
            duration_ms=int((time.time() - synth_start) * 1000)
        ))
    else:
        # Structured fallback if no LLM key configured
        parts = []
        if effective_lang == "hi":
            if state.triage_data:
                parts.append(f"**🩺 संभावित लक्षण मूल्यांकन:** {state.triage_data.get('urgency_badge', '🟢 सामान्य स्वास्थ्य')}")
                parts.append(f"**📋 अगले कदम:** {state.triage_data.get('recommended_action', 'पर्याप्त आराम करें और 24-48 घंटे में डॉक्टर से परामर्श लें।')}")
                t_level = state.triage_data.get("triage_level", "HOME_CARE")
                if t_level == "EMERGENCY_CARE":
                    parts.append("\n**💊 दवाइयां एवं राहत (भारत):**\n• ⚠️ *स्व-दवा से बचें:* डॉक्टर के परीक्षण से पहले कोई दवा न लें।\n• *अस्पताल में:* आपातकालीन टीम द्वारा आईवी ड्रिप व उपचार दिया जाएगा।")
                    parts.append("\n**🚨 तुरंत 108 / 112 पर कॉल करें यदि:** सांस लेने में तकलीफ, SpO2 < 92%, या बेहोशी हो।")
                else:
                    parts.append("\n**💊 संभावित दवाइयां एवं राहत (भारत):**\n• *Dolo 650 (पैरासिटामोल 650mg):* बुखार/दर्द के लिए भोजन के बाद पानी से 1 गोली (दिन में अधिकतम 3 बार)।\n• *Electral ORS:* 1 लीटर पानी में 1 पैकेट घोलकर दिनभर घूंट-घूंट पिएं।\n• *Pan-40:* एसिडिटी होने पर सुबह खाली पेट नाश्ते से 30 मिनट पहले 1 गोली।")
                    parts.append("\n**🛡️ निवारक उपाय एवं देखभाल:**\n• पर्याप्त आराम करें, तरल पदार्थ पिएं, और तेज बुखार होने पर माथे पर सामान्य पानी की पट्टी रखें।")
                    parts.append("\n**🚨 डॉक्टर को दिखाएं / 108 पर कॉल करें यदि:** बुखार 103°F से अधिक हो, सांस फूलने लगे या गर्दन में अकड़न हो।")
                parts.append("\n⚠️ *एआई सूचना:* यह केवल शैक्षिक मार्गदर्शन है। किसी भी दवा से पहले डॉक्टर से परामर्श अवश्य लें।")
            else:
                parts.append("संजीवनी एआई द्वारा आपके स्वास्थ्य का विश्लेषण किया गया है। कृपया आराम करें और आवश्यकता पड़ने पर चिकित्सक से परामर्श लें।\n\n⚠️ *एआई सूचना:* यह केवल मार्गदर्शन है।")
        else:
            if state.vaccination_data:
                v_data = state.vaccination_data
                parts.append(f"**💉 UIP Vaccination Status:** Next Due: **{v_data.get('next_vaccine_due')}** ({v_data.get('next_due_date')})")
                parts.append(f"• **National Immunization Progress:** {v_data.get('uip_compliance_pct', 100)}% UIP Milestones Completed")

            if state.triage_data:
                parts.append(f"**🩺 Triage Assessment:** {state.triage_data.get('urgency_badge')}")
                parts.append(f"**📋 Next Steps:** {state.triage_data.get('recommended_action')}")
                if state.triage_data.get("recommended_specialist"):
                    parts.append(f"• **Recommended Specialist:** {state.triage_data['recommended_specialist']}")

                t_level = state.triage_data.get("triage_level", "HOME_CARE")
                if t_level == "EMERGENCY_CARE":
                    parts.append("\n**💊 Medications & Relief (India):**\n• ⚠️ *Strictly Withhold Self-Medication:* Do not take painkillers or anti-emetics before hospital examination (masks neurological & abdominal signs).\n• *At Hospital:* IV fluids and emergency targeted therapy will be administered.")
                    parts.append("\n**🚨 Call 108 / 112 Immediately If:** Severe chest pain, shortness of breath, SpO2 < 92%, or altered consciousness.")
                else:
                    parts.append("\n**💊 Probable Medications & Relief (India):**\n• *Dolo 650 (Paracetamol 650mg):* 1 tablet after meals with water for fever/pain (max 3/day).\n• *Electral ORS:* 1 packet in 1L clean drinking water; sip throughout the day for active hydration.\n• *Pan-40 (Pantoprazole 40mg):* 1 tablet 30 minutes before breakfast on empty stomach if gastric acidity occurs.")
                    parts.append("\n**🛡️ Preventive Measures & Home Care:**\n• Ensure complete physical rest, drink 2-3L fluids/ORS, eat light food, and use lukewarm sponge baths for high fever.")
                    parts.append("\n**🚨 Seek Emergency Care / Call 108 If:** Fever exceeds 103°F, shortness of breath develops, or neck stiffness occurs.")
                parts.append("\n⚠️ *AI Disclaimer:* Educational AI guidance only. Please consult a qualified physician or clinic for diagnosis and prescription.")

            if state.drug_check and state.drug_check.get("detected_medications"):
                meds = ", ".join(state.drug_check["detected_medications"])
                parts.append(f"\n**Medication Scan:** Detected {meds}")

        state.final_response = "\n\n".join(parts) if parts else "Health assessment completed by Synapse-OS Swarm."

    state.suggested_actions = [
        "View 3D Digital Health Twin",
        "Generate Verifiable Health Passport (QR)",
        "Check Universal Immunization Schedule",
        "View District Outbreak Early Warning"
    ]

    return state

